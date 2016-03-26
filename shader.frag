#version 400
in vec2 UV;

layout (location = 0) out vec4 uFragColor;

void main() {
   uFragColor = vec4(UV, 0.f, 1.f);
}
