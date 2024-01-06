#version 450

layout (location = 0) in vec3 fragColor;
/* layout (location = 1) in vec2 fragTexCoord; */
layout (location = 0) out vec4 outColor;

/* layout (binding = 1) uniform sampler2D texSampler; */

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float l = length(coord) * 2;
    if (l > 1) {
        outColor = vec4(fragColor.rgb, 0.0);
    } else {
        outColor = vec4(fragColor, sqrt(1 - l * l));
    }
}