//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
	precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m02; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m02; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m02; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
};

struct vec3 {
	float x, y, z;

	vec3() {}

	vec3(float argx, float argy, float argz) {
		x = argx;
		y = argy;
		z = argz;
	}

	vec3 operator+(const vec3 arg) {
		vec3 v;
		v.x = this->x + arg.x;
		v.y = this->y + arg.y;
		v.z = this->z + arg.z;
		return v;
	}
	vec3 operator-(const vec3 arg) {
		vec3 v;
		v.x = this->x - arg.x;
		v.y = this->y - arg.y;
		v.z = this->z - arg.z;
		return v;
	}
	vec3 operator/(const float arg) {
		vec3 v;
		v.x = this->x / arg;
		v.y = this->y / arg;
		v.z = this->z / arg;
		return v;
	}
	vec3 operator*(const float arg) {
		vec3 v;
		v.x = this->x * arg;
		v.y = this->y * arg;
		v.z = this->z * arg;
		return v;
	}
	vec3& operator=(const vec3 arg) {
		this->x = arg.x;
		this->y = arg.y;
		this->z = arg.z;
		return *this;
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
public:
	Triangle() {
		Animate(0);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[1000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		//GLuint vbo;	// vertex/index buffer object
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		if (nVertices >= 200) return;

		//vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = cX;
		vertexData[5 * nVertices + 1] = cY;
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 1; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void removeAll() {
		nVertices = 0;
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

class ControlPoint {
	vec3 pos;
	float t;
	float time;
public:
	ControlPoint() {}

	ControlPoint(float argt, float x, float y, float z, float time) {
		pos = vec3(x, y, z);
		t = argt;
		this->time = time;
	}

	vec3 getPos() {
		return pos;
	}

	float getTimeValue() {
		return t;
	}

	float getCPTime() {
		return time;
	}

	void setPos(vec3 argv) {
		pos = argv;
	}

	void setTimeValue(float argt) {
		t = argt;
	}

	ControlPoint& operator=(ControlPoint arg) {
		this->pos = arg.getPos();
		this->t = arg.getTimeValue();
		return *this;
	}
};

class BezierSurface {
	std::vector<std::vector<vec3>> controlMesh;
	unsigned int vao; // vertex array object id
	float sx, sy;
	float wTx, wTy;

	float B(int i, float t) {
		int n = controlMesh[i].size() - 1; // n deg polynomial = n + 1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) {
			choose *= (float)(n - j + 1) / j;
		}
		return choose * pow(t, i) * pow(1 - t, n - i);
	}

public:
	BezierSurface() {

	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
											   //static float vertexCoords[6] = coordArray;	// vertex data on the CPU
		static float vertexCoords[100000];
		GenerateMesh();
		int n = 0;
		vec3 v;
		vec3 downleft;
		vec3 upperleft;
		vec3 downright;
		vec3 upperright;
		float dt = 1.0f / 5.0f / 10.0f;
		for (float i = 0.0f; i < 1.0f; i += dt) {
			for (float j = 0.0f; j < 1.0f; j += dt) {
				downleft = r(i, j);
				upperleft = r(i, j + dt);
				downright = r(i + dt, j);
				upperright = r(i + dt, j + dt);

				//first triangle
				vertexCoords[n] = downleft.x;
				vertexCoords[n + 1] = downleft.y;
				vertexCoords[n + 2] = downleft.z;

				vertexCoords[n + 3] = upperleft.x;
				vertexCoords[n + 4] = upperleft.y;
				vertexCoords[n + 5] = upperleft.z;

				vertexCoords[n + 6] = downright.x;
				vertexCoords[n + 7] = downright.y;
				vertexCoords[n + 8] = downright.z;
				//second
				vertexCoords[n + 9] = upperright.x;
				vertexCoords[n + 10] = upperright.y;
				vertexCoords[n + 11] = upperright.z;

				vertexCoords[n + 12] = downright.x;
				vertexCoords[n + 13] = downright.y;
				vertexCoords[n + 14] = downright.z;

				vertexCoords[n + 15] = upperleft.x;
				vertexCoords[n + 16] = upperleft.y;
				vertexCoords[n + 17] = upperleft.z;

				n += 18;
			}
		}

		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords),  // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0
		glVertexAttribPointer(0,			// Attribute Array 0
			3, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[100000];	// vertex data on the CPU
		n = 0;
		for (float i = 0.0f; i < 1.0f; i += dt) {
			for (float j = 0.0f; j < 1.0f; j += dt) {
				downleft = r(i, j);
				upperleft = r(i, j + dt);
				downright = r(i + dt, j);
				upperright = r(i + dt, j + dt);

				float red = 0.15f;
				float green = 0.2f;
				//first triangle
				vertexColors[n] = downleft.z / 5;
				vertexColors[n + 1] = (5 - downleft.z) / 5;
				vertexColors[n + 2] = 0.0f;

				vertexColors[n + 3] = upperleft.z / 5;
				vertexColors[n + 4] = (5 - upperleft.z) / 5;
				vertexColors[n + 5] = 0.0f;

				vertexColors[n + 6] = downright.z / 5;
				vertexColors[n + 7] = (5 - downright.z) / 5;
				vertexColors[n + 8] = 0.0f;

				//second triangle
				vertexColors[n + 9] = upperright.z / 5;
				vertexColors[n + 10] = (5 - upperright.z) / 5;
				vertexColors[n + 11] = 0.0f;

				vertexColors[n + 12] = downright.z / 5;
				vertexColors[n + 13] = (5 - downright.z) / 5;
				vertexColors[n + 14] = 0.0f;

				vertexColors[n + 15] = upperleft.z / 5;
				vertexColors[n + 16] = (5 - upperleft.z) / 5;
				vertexColors[n + 17] = 0.0f;

				n += 18;
			}
		}

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Draw() {
		wTx = 0;
		wTy = 0;
		sx = 1;
		sy = 1;
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 100000);	// draw a single triangle with vertices defined in vao

	}

	void GenerateMesh() {
		//height matrix from 0 to 5
		float heightMatrix[6][6] = {
			{ 0, 1, 2, 3, 4, 5 },
			{ 0, 2, 1, 1, 0, 4 },
			{ 2, 3, 5, 5, 0, 1 },
			{ 3, 4, 5, 5, 3, 2 },
			{ 4, 5, 4, 3, 2, 1 },
			{ 5, 4, 0, 2, 1, 5 }
		};
		//filling up the mesh
		int a = 0;
		int b = 0;
		for (int i = -10; i <= 10; i += 4) {
			std::vector<vec3> xArray;
			for (int j = -10; j <= 10; j += 4) {
				xArray.push_back(vec3(i, j, heightMatrix[a][b]));
				b++;
			}
			controlMesh.push_back(xArray);
			b = 0;
			a++;
		}
	}

	vec3 r(float u, float v) {
		vec3 rr(0, 0, 0);
		for (int i = 0; i < controlMesh.size(); i++) {
			for (int j = 0; j < controlMesh[i].size(); j++) {
				rr = rr + controlMesh[i][j] * B(i, u) * B(j, v);
			}
		}
		return rr;
	}

};

class LagrangeCurve {
	std::vector<ControlPoint> cps;
	float totalTime;
	float length;

	float L(int i, float t) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++) {
			if (j != i)
				Li *= (t - cps[j].getTimeValue()) / (cps[i].getTimeValue() - cps[j].getTimeValue());
		}
		return Li;
	}

public: 

	LagrangeCurve() { length = 0; }

	void AddControlPoint(float time, float cx, float cy,  LineStrip& l, BezierSurface& b) {
		vec4 wVertex = vec4(cx, cy, 0, 1) * camera.Pinv() * camera.Vinv();
		float x = wVertex.v[0];
		float y = wVertex.v[1];
		float t = 0;

		ControlPoint cp = ControlPoint(cps.size(), x, y, 0, time);
		cps.push_back(cp);

		totalTime = cps.back().getCPTime() - cps.front().getCPTime();

		l.removeAll();
		float dt = 0.1f;
		if (cps.size() < 2) {
			for (int i = 0; i < cps.size(); i++) {
				l.AddPoint(cps[i].getPos().x, cps[i].getPos().y);
			}
		}
		else {
			for (float i = 0.0f; i <= cps.size() - 0.99f; i += dt) {
				vec3 rr = r(i);
				l.AddPoint(rr.x, rr.y);
				if (i > 0.0f) {
					vec3 prerr = r(i - dt);
					vec3 br = b.r((rr.x + 10) / 20, (rr.y + 10) / 20);
					vec3 prebr = b.r((prerr.x + 10) / 20, (prerr.y + 10) / 20);
					float d = sqrtf(pow((br.x - prebr.x), 2) + pow((br.y - prebr.y), 2) + pow((br.z * 2 - prebr.z * 2), 2));
					length += d;
				}
			}
		}
		printf("Total distance: %f m \n", length * 50);

		
	}

	vec3 r(float t) {
		vec3 rr(0, 0, 0);
		for (int i = 0; i < cps.size(); i++) {
			rr = rr + (cps[i].getPos() * L(i, t));
		}
		return rr;
	}

	std::vector<ControlPoint> getCPs() {
		return cps;
	}
};

class BezierCurve {
	std::vector<ControlPoint> cps;

	float B(int i, float t) {
		int n = cps.size() - 1; // n deg polynomial = n + 1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) {
			choose *= (float) (n - j + 1) / j;
		}
		return choose * pow(t, i) * pow(1 - t, n - i);
	}

public:
	void AddControlPoint(float time, float cx, float cy, LineStrip& l) {
		vec4 wVertex = vec4(cx, cy, 0, 1) * camera.Pinv() * camera.Vinv();
		float x = wVertex.v[0];
		float y = wVertex.v[1];
		ControlPoint cp = ControlPoint(time, x, y, 0, time);
		cps.push_back(cp);

		l.removeAll();			
		if (cps.size() <= 2) {
			for (int i = 0; i < cps.size(); i++) {
				l.AddPoint(cps[i].getPos().x, cps[i].getPos().y);
			}
		}
		else {

			float dt = 0.01f;
			for (float i = 0.0f; i <= 1.0f; i+= dt) {
				vec3 rr = r(i);
				l.AddPoint(rr.x, rr.y);
			}
		}
	}

	vec3 r(float t) {
		vec3 rr(0, 0, 0);
		for (int i = 0; i < cps.size(); i++) {
			rr = rr + (cps[i].getPos() * B(i, t));
			//printf("i2: %d\n", i);
		}
		return rr;
	}
};

class Bicycle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	int state;
	float time;
	float iterator;
	float preTime;
	float dt;
public:
	Bicycle() {
		state = false;
	}

	int getState() {
		return state;
	}

	void toggle(int arg, float time) {
		iterator = 0;
		state = arg;
		preTime = glutGet(GLUT_ELAPSED_TIME);
		dt = 0;
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		vec3 p1 = vec3(0.0f, 0.0f, 10.0f);
		vec3 p2 = vec3(-0.4f, -0.4f, 10.0f);
		vec3 p3 = vec3(0.4f, -0.4f, 10.0f);
		vec3 p4 = vec3(0.0f, 0.5f, 10.0f);
		static float vertexCoords[] = {
			p1.x, p1.y, p1.z,
			p2.x, p2.y, p2.z,
			p4.x, p4.y, p4.z,

			p1.x, p1.y, p1.z,
			p3.x, p3.y, p3.z,
			p4.x, p4.y, p4.z,
		};
		// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			3, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = {
			0, 0, 0,
			0, 0, 0,
			0, 0, 0,

			0, 0, 0,
			0, 0, 0,
			0, 0, 0,
		};	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t, LagrangeCurve *l) {
		if (state) {
			sx = 1;
			sy = 1;
			std::vector<ControlPoint> cps = l->getCPs();
			if (iterator < cps.size() - (dt + 1.0f)) {
				if (cps.size() > 1) {
					/*if (glutGet(GLUT_ELAPSED_TIME) % 10 == 0) {
						vec3 rr = l->r(iterator);
						wTx = rr.x;
						wTy = rr.y;
						printf("wTx: %f\t wTy: %f\n", wTx, wTy);
						if (iterator <= cps.size() - 0.99f)
							iterator += dt;
					}*/

					if (dt <= 0.00001000) {
						if (glutGet(GLUT_ELAPSED_TIME) != preTime) {
							dt = 1 / ((cps[1].getCPTime() - cps[0].getCPTime()) * 1000);
							vec3 rr = l->r(iterator);
							wTx = rr.x;
							wTy = rr.y;
							//printf("dt: %f\n", dt);
							iterator += dt;
						}
						preTime = glutGet(GLUT_ELAPSED_TIME);
					}
					else {
						float dx = cps[ceil(iterator)].getCPTime() - cps[ceil(iterator) - 1].getCPTime();
						dt = 1 / (dx * 1000);
						if (glutGet(GLUT_ELAPSED_TIME) != preTime) {
							vec3 rr = l->r(iterator);
							wTx = rr.x;
							wTy = rr.y;
							//printf("dt: %f\n", dx);
							iterator += dt;
						}
						preTime = glutGet(GLUT_ELAPSED_TIME);
					}
				}
				else { this->toggle(false, 0); }
			}
		}
	}

		void Draw() {
			mat4 Mscale(sx, 0, 0, 0,
				0, sy, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 1); // model matrix

			mat4 Mtranslate(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 0, 0,
				wTx, wTy, 0, 1); // model matrix

			mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

			// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
			glDrawArrays(GL_TRIANGLES, 0, 6);	// draw a single triangle with vertices defined in vao
		}
};

// The virtual world: collection of two objects
Triangle triangle;
LineStrip lineStrip;
LagrangeCurve lagrangeCurve;
BezierCurve bezierCurve;
LineStrip lineStripBezier;
BezierSurface bezierSurface;
Bicycle bicycle;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	//triangle.Create();
	bezierSurface.Create();
	lineStrip.Create();
	lineStripBezier.Create();
	bicycle.Create();


	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	//triangle.Draw();
	bezierSurface.Draw();
	lineStrip.Draw();
	lineStripBezier.Draw();
	bicycle.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float sec = time / 1000.0f;
	if (key == ' ') {
		if (bicycle.getState())
			bicycle.toggle(false, time);
		else
			bicycle.toggle(true, 0);
	}
		
	glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float sec = time / 1000.0f;
	float cX;
	float cY;
	//lagrange/linestrip
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		cY = 1.0f - 2.0f * pY / windowHeight;
		//lineStrip.AddPoint(cX, cY);
		lagrangeCurve.AddControlPoint(sec, cX, cY, lineStrip, bezierSurface);
		glutPostRedisplay();     // redraw
	}

	//bezier
	if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		cY = 1.0f - 2.0f * pY / windowHeight;
		bezierCurve.AddControlPoint(sec, cX, cY, lineStripBezier);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	//triangle.Animate(sec);					// animate the triangle object
	bicycle.Animate(sec, &lagrangeCurve);
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

