#version 400
in vec3 col;

layout (location = 0) out vec4 uFragColor;

void main() {
   uFragColor = vec4(col, 1.f);
}
