#version 430

layout(location = 0) in vec3 vert_loc;

void main()
{
	gl_Position.xyz = vert_loc;
	gl_Position.w = 1.f;
}
