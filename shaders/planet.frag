#version 430
layout( location = 0 ) out vec4 FragColor;

in vec3 Position;
in vec3 vPos;
in vec3 Normal;
in vec2 UV;
in vec3 ShadowUV;
in float Altitude;

uniform vec3 sunPos;
uniform sampler2D tex;

//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
// 

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v) {
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i); 
    vec4 p = permute(permute(permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;

    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
        dot(p2,x2), dot(p3,x3) ) );
}

//  <https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83>
#define NUM_OCTAVES 5
float fbm(vec3 x, float freq, float amp, float lacunarity, float gain) {
    float sum = 0.0f;
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        sum += amp * snoise(x * freq);
        freq *= lacunarity;
        amp *= gain;
    }
    return sum;
}

void main () {
    float oceanHeight = 0.65f;
    float sandHeight = oceanHeight + 0.015f;
    float snowHeight = oceanHeight + 0.13f;

	vec3 X = dFdx(Position);
	vec3 Y = dFdy(Position);
	vec3 normal = normalize(cross(X,Y));

///////////////////////////////////////////////////////////////////////////
// Color

    float noise = clamp(abs(fbm(vPos, 1.0f, 0.5f, 1.8715f, 0.5f)), 0.0f, 0.5f);
    float snowNoise = clamp(fbm(vPos, 1.2f, 0.2f, 1.8715f, 0.5f), -0.01f, 0.01f);

    float red = step(sandHeight, Altitude) *  step(Altitude, snowHeight  + snowNoise) * noise;
    float blue = step(Altitude, oceanHeight) * (1.f - (oceanHeight - Altitude));
    float green = step(sandHeight, Altitude) *  step(Altitude, snowHeight + snowNoise) * (1 - noise);
    vec3 sandColor = vec3(1.f, 0.878f, 0.619f) * step(oceanHeight, Altitude) * step(Altitude, sandHeight);

    float snowHeightNoise = snowHeight + snowNoise;
    vec3 snowColor = vec3(1.f, 1.f, 1.f);

    vec3 color = mix(vec3(red, green, pow(blue, 5.f)), snowColor, vec3(smoothstep(snowHeight - 0.04, snowHeightNoise, Altitude)));
    color += sandColor;
    clamp(color, vec3(0.f), vec3(1.0f));

    ////////////////////////////////////////////////////////////////////////////
    // Lighting
    vec3 lightDir = normalize(sunPos - Position);
    vec3 viewDir  = normalize(-Position);

    // Shadow sample
    vec2 shadowValue = texture(tex, vec2(1.0 - ShadowUV.x, 1.0 - ShadowUV.y)).xy;
    float shadowAmt = (shadowValue.x > ShadowUV.z + 0.001) ? 1.0 : 0.0;

    // city lights
    float cityLightNoise = max(clamp(dot(normal, -lightDir), 0.0f, 1.0f), 0.5*shadowAmt) * smoothstep(0.0, 0.2, snoise(vPos.xyz * 1)) * smoothstep(0.2, 0.4, snoise(vPos.xyz * 3)) * clamp(snoise(vPos.xyz * 15) + 0.2, 0.0, 1.0);
    vec3 lightPolutionColor = vec3(1.0f, 0.98f, 0.914f);
    vec3 lightPolutionColor2 = vec3(0.961f, 0.518f, 0.29f);
    vec3 emissive = mix(lightPolutionColor2, lightPolutionColor2, cityLightNoise) * step(sandHeight, Altitude) *  step(Altitude, snowHeight - 0.06) * cityLightNoise;
    clamp(color, vec3(0.f), vec3(1.0f));

    //Diffuse part-----------
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * color;

    //specular part-------------
    const float shininess = 8.0;
    vec3 H = normalize(lightDir + viewDir);
    float NdH = max(dot(H, normal), 0.0);
    float spec = pow(NdH, shininess);
    vec3 specularColor = vec3(0.2f, 0.2f, 0.2f);
    vec3 sunColor = vec3(0.9321f, 0.97f, 0.7039f);
    vec3 specular = mix(vec3(0.0), spec * sunColor, step(Altitude, oceanHeight));

    // Ambient-------------
    vec3 ambient = 0.08f * color;

    vec3 resultLight = ambient + (1.0 - shadowAmt) * diffuse + specular * 0.8 + emissive;

    FragColor = vec4(pow(resultLight, vec3(2.2)), 1.0f);
}
