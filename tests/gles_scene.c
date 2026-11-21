/* Scene probe for the render-path model (surfaceless EGL, GLES 3.0 through Mesa's open-source asahi driver).
 * Renders one described scene into an offscreen target, a few times (one flush per frame, so each frame is one
 * render command), and writes the last frame's RGBA8 pixels (glReadPixels, bottom row first) to OUT.
 * Run it under tests/render_trace.c to capture the command Mesa builds; scripts/render_scene.py writes scene files.
 *
 * usage: gles_scene SCENE OUT [frames]
 * SCENE lines (one item each, # comments):
 *   size W H | samples N (0 = single sample) | clear R G B A | cleardepth Z
 *   depth off|never|less|equal|lequal|greater|notequal|gequal|always   (depth test; depthwrite 0|1, default 1)
 *   stencil off|FUNC REF READMASK WRITEMASK SFAIL ZFAIL ZPASS  (ops: keep zero replace incr decr invert incrwrap decrwrap)
 *   cull none|back|front|both | frontface ccw|cw | scissor X Y W H | blend off|alpha|add
 *   shader smooth|flat|discard|fragz|const   (fragment: colour varying; flat; discard where colour.r < 0.5; gl_FragDepth = colour.g; constant)
 *   prim triangles|strip|fan | indexed 0|1 | restart 0|1 (primitive restart index 0xffff)
 *   v X Y Z W R G B A        (a vertex, clip space; draws use them in order)
 *   i N                      (an index, for indexed draws; 65535 = restart)
 *   draw FIRST COUNT [INSTANCES]   (a draw call over vertices / indices; instance k shifts x by 0.05 * k)
 */
#define _POSIX_C_SOURCE 200809L
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EGL_FUNCS(X) \
 X(PFNEGLGETPLATFORMDISPLAYPROC, eglGetPlatformDisplay) X(PFNEGLINITIALIZEPROC, eglInitialize) \
 X(PFNEGLBINDAPIPROC, eglBindAPI) X(PFNEGLCHOOSECONFIGPROC, eglChooseConfig) \
 X(PFNEGLCREATECONTEXTPROC, eglCreateContext) X(PFNEGLMAKECURRENTPROC, eglMakeCurrent)
#define GL_FUNCS(X) \
 X(PFNGLCREATESHADERPROC, glCreateShader) X(PFNGLSHADERSOURCEPROC, glShaderSource) \
 X(PFNGLCOMPILESHADERPROC, glCompileShader) X(PFNGLGETSHADERIVPROC, glGetShaderiv) \
 X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog) X(PFNGLCREATEPROGRAMPROC, glCreateProgram) \
 X(PFNGLATTACHSHADERPROC, glAttachShader) X(PFNGLLINKPROGRAMPROC, glLinkProgram) \
 X(PFNGLUSEPROGRAMPROC, glUseProgram) X(PFNGLGENBUFFERSPROC, glGenBuffers) \
 X(PFNGLBINDBUFFERPROC, glBindBuffer) X(PFNGLBUFFERDATAPROC, glBufferData) \
 X(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays) X(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray) \
 X(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer) \
 X(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray) \
 X(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers) X(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer) \
 X(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers) X(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer) \
 X(PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC, glRenderbufferStorageMultisample) \
 X(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer) \
 X(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus) \
 X(PFNGLVIEWPORTPROC, glViewport) X(PFNGLCLEARCOLORPROC, glClearColor) X(PFNGLCLEARPROC, glClear) \
 X(PFNGLCLEARDEPTHFPROC, glClearDepthf) X(PFNGLCLEARSTENCILPROC, glClearStencil) \
 X(PFNGLDRAWARRAYSINSTANCEDPROC, glDrawArraysInstanced) X(PFNGLDRAWELEMENTSINSTANCEDPROC, glDrawElementsInstanced) \
 X(PFNGLENABLEPROC, glEnable) X(PFNGLDISABLEPROC, glDisable) X(PFNGLDEPTHFUNCPROC, glDepthFunc) X(PFNGLDEPTHMASKPROC, glDepthMask) \
 X(PFNGLSTENCILFUNCPROC, glStencilFunc) X(PFNGLSTENCILOPPROC, glStencilOp) X(PFNGLSTENCILMASKPROC, glStencilMask) \
 X(PFNGLCULLFACEPROC, glCullFace) X(PFNGLFRONTFACEPROC, glFrontFace) X(PFNGLSCISSORPROC, glScissor) \
 X(PFNGLBLENDFUNCPROC, glBlendFunc) X(PFNGLFINISHPROC, glFinish) X(PFNGLFLUSHPROC, glFlush) \
 X(PFNGLREADPIXELSPROC, glReadPixels) X(PFNGLGETERRORPROC, glGetError)
#define DECL(t, n) static t p_##n;
EGL_FUNCS(DECL)
GL_FUNCS(DECL)

static void fail(const char *s) { fprintf(stderr, "FAIL %s\n", s); exit(1); }
static GLuint compile(GLenum type, const char *src) {
    GLuint s = p_glCreateShader(type); p_glShaderSource(s, 1, &src, NULL); p_glCompileShader(s);
    GLint ok; p_glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; p_glGetShaderInfoLog(s, sizeof log, NULL, log); fprintf(stderr, "%s\n", log); fail("compile"); }
    return s;
}
static GLenum zfunc(const char *s) {
    const char *n[] = {"never", "less", "equal", "lequal", "greater", "notequal", "gequal", "always"};
    for (int i = 0; i < 8; i++) if (!strcmp(s, n[i])) return GL_NEVER + i;
    fail("depth func"); return 0;
}
static GLenum sop(const char *s) {
    if (!strcmp(s, "keep")) return GL_KEEP; if (!strcmp(s, "zero")) return GL_ZERO; if (!strcmp(s, "replace")) return GL_REPLACE;
    if (!strcmp(s, "incr")) return GL_INCR; if (!strcmp(s, "decr")) return GL_DECR; if (!strcmp(s, "invert")) return GL_INVERT;
    if (!strcmp(s, "incrwrap")) return GL_INCR_WRAP; if (!strcmp(s, "decrwrap")) return GL_DECR_WRAP;
    fail("stencil op"); return 0;
}

#define MAXV 65536
static float verts[MAXV][8]; static int nverts;
static uint16_t idx[MAXV]; static int nidx;
static struct { int first, count, inst; } draws[1024]; static int ndraws;

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s SCENE OUT [frames]\n", argv[0]); return 2; }
    int frames = argc > 3 ? atoi(argv[3]) : 2;
    int W = 64, H = 64, samples = 0, indexed = 0, restart = 0, depthwrite = 1, have_depth = 0, have_stencil = 0;
    float clear[4] = {0, 0, 0, 1}, cleardepth = 1.0f;
    char depth[32] = "off", cull[16] = "none", ff[8] = "ccw", blend[16] = "off", shader[16] = "smooth", prim[16] = "triangles";
    char st[8][16] = {{0}}; int sten = 0; int scissor[4] = {-1, -1, -1, -1};
    FILE *f = fopen(argv[1], "r"); if (!f) fail("scene");
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *h = strchr(line, '#'); if (h) *h = 0;
        char k[32]; if (sscanf(line, "%31s", k) != 1) continue;
        float *v;
        if (!strcmp(k, "size")) sscanf(line, "%*s %d %d", &W, &H);
        else if (!strcmp(k, "samples")) sscanf(line, "%*s %d", &samples);
        else if (!strcmp(k, "clear")) sscanf(line, "%*s %f %f %f %f", &clear[0], &clear[1], &clear[2], &clear[3]);
        else if (!strcmp(k, "cleardepth")) sscanf(line, "%*s %f", &cleardepth);
        else if (!strcmp(k, "depth")) sscanf(line, "%*s %31s", depth);
        else if (!strcmp(k, "depthwrite")) sscanf(line, "%*s %d", &depthwrite);
        else if (!strcmp(k, "stencil")) { sten = sscanf(line, "%*s %15s %15s %15s %15s %15s %15s %15s", st[0], st[1], st[2], st[3], st[4], st[5], st[6]); if (!strcmp(st[0], "off")) sten = 0; }
        else if (!strcmp(k, "cull")) sscanf(line, "%*s %15s", cull);
        else if (!strcmp(k, "frontface")) sscanf(line, "%*s %7s", ff);
        else if (!strcmp(k, "scissor")) sscanf(line, "%*s %d %d %d %d", &scissor[0], &scissor[1], &scissor[2], &scissor[3]);
        else if (!strcmp(k, "blend")) sscanf(line, "%*s %15s", blend);
        else if (!strcmp(k, "shader")) sscanf(line, "%*s %15s", shader);
        else if (!strcmp(k, "prim")) sscanf(line, "%*s %15s", prim);
        else if (!strcmp(k, "indexed")) sscanf(line, "%*s %d", &indexed);
        else if (!strcmp(k, "restart")) sscanf(line, "%*s %d", &restart);
        else if (!strcmp(k, "v")) { if (nverts >= MAXV) fail("verts"); v = verts[nverts++];
            if (sscanf(line, "%*s %f %f %f %f %f %f %f %f", v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7) != 8) fail("v line"); }
        else if (!strcmp(k, "i")) { int x; sscanf(line, "%*s %d", &x); if (nidx >= MAXV) fail("idx"); idx[nidx++] = (uint16_t)x; }
        else if (!strcmp(k, "draw")) { int a = 0, b = 0, c = 1; sscanf(line, "%*s %d %d %d", &a, &b, &c); if (ndraws >= 1024) fail("draws");
            draws[ndraws].first = a; draws[ndraws].count = b; draws[ndraws++].inst = c; }
        else fail(k);
    }
    fclose(f);
    have_depth = strcmp(depth, "off") != 0 || !strcmp(shader, "fragz"); have_stencil = sten > 0;

    void *egl = dlopen("libEGL.so.1", RTLD_NOW); if (!egl) fail("libEGL");
    PFNEGLGETPROCADDRESSPROC gpa = (PFNEGLGETPROCADDRESSPROC)dlsym(egl, "eglGetProcAddress");
#define LOADE(t, n) p_##n = (t)gpa(#n);
#define LOADG(t, n) p_##n = (t)gpa(#n);
    EGL_FUNCS(LOADE) GL_FUNCS(LOADG)
    EGLDisplay d = p_eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    EGLint maj, min; if (!p_eglInitialize(d, &maj, &min)) fail("eglInitialize");
    p_eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ca[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, 0, EGL_NONE};
    EGLConfig cfg; EGLint n; if (!p_eglChooseConfig(d, ca, &cfg, 1, &n) || !n) fail("config");
    EGLint cx[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE};
    EGLContext c = p_eglCreateContext(d, cfg, EGL_NO_CONTEXT, cx); if (!c) fail("context");
    if (!p_eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) fail("makecurrent");

    const char *q = !strcmp(shader, "flat") ? "flat " : "";
    char vs[1024], fs[1024];
    snprintf(vs, sizeof vs, "#version 300 es\nin vec4 p;\nin vec4 col;\nuniform float dx;\n%sout vec4 c;\n"
             "void main(){ c = col; gl_Position = p + vec4(dx * float(gl_InstanceID), 0.0, 0.0, 0.0); }\n", q);
    char alub[512];
    if (!strncmp(shader, "alu", 3)) {   /* aluN: N dependent multiply-adds per fragment on the colour (loop count is a uniform: not unrolled) */
        snprintf(alub, sizeof alub, "vec4 a = c; for (int i = 0; i < n; i++) { a = a * a * 0.5 + 0.25; } o = a;");
    }
    const char *body = !strncmp(shader, "alu", 3) ? alub : !strcmp(shader, "discard") ? "if (c.r < 0.5) discard; o = c;"
                     : !strcmp(shader, "fragz") ? "o = c; gl_FragDepth = c.g;"
                     : !strcmp(shader, "const") ? "o = vec4(0.25, 0.5, 0.75, 1.0);" : "o = c;";
    snprintf(fs, sizeof fs, "#version 300 es\nprecision highp float;\nuniform int n;\n%sin vec4 c;\nout vec4 o;\nvoid main(){ %s }\n", q, body);
    GLuint prog = p_glCreateProgram();
    p_glAttachShader(prog, compile(GL_VERTEX_SHADER, vs)); p_glAttachShader(prog, compile(GL_FRAGMENT_SHADER, fs));
    p_glLinkProgram(prog); p_glUseProgram(prog);
    { typedef GLint (*GUL)(GLuint, const char *); typedef void (*U1F)(GLint, float);
      GUL gul = (GUL)gpa("glGetUniformLocation"); U1F u1 = (U1F)gpa("glUniform1f"); u1(gul(prog, "dx"), 0.05f);
      typedef void (*U1I)(GLint, int); U1I ui = (U1I)gpa("glUniform1i");
      if (!strncmp(shader, "alu", 3)) ui(gul(prog, "n"), atoi(shader + 3)); }

    GLuint vao, vbo, ibo; p_glGenVertexArrays(1, &vao); p_glBindVertexArray(vao);
    p_glGenBuffers(1, &vbo); p_glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p_glBufferData(GL_ARRAY_BUFFER, (long)nverts * 32, verts, GL_STATIC_DRAW);
    typedef GLint (*GAL)(GLuint, const char *); GAL gal = (GAL)gpa("glGetAttribLocation");
    GLint lp = gal(prog, "p"), lc = gal(prog, "col");
    p_glVertexAttribPointer(lp, 4, GL_FLOAT, GL_FALSE, 32, 0); p_glEnableVertexAttribArray(lp);
    if (lc >= 0) { p_glVertexAttribPointer(lc, 4, GL_FLOAT, GL_FALSE, 32, (void *)16); p_glEnableVertexAttribArray(lc); }
    if (indexed) { p_glGenBuffers(1, &ibo); p_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); p_glBufferData(GL_ELEMENT_ARRAY_BUFFER, (long)nidx * 2, idx, GL_STATIC_DRAW); }

    GLuint fbo, rb; p_glGenFramebuffers(1, &fbo); p_glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    p_glGenRenderbuffers(1, &rb); p_glBindRenderbuffer(GL_RENDERBUFFER, rb);
    p_glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, W, H);
    p_glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    if (have_depth || have_stencil) {
        GLuint db; p_glGenRenderbuffers(1, &db); p_glBindRenderbuffer(GL_RENDERBUFFER, db);
        p_glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, have_stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32F, W, H);
        p_glFramebufferRenderbuffer(GL_FRAMEBUFFER, have_stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, db);
    }
    if (p_glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) fail("fbo incomplete");
    p_glViewport(0, 0, W, H);
    if (have_depth) { p_glEnable(GL_DEPTH_TEST); p_glDepthFunc(strcmp(depth, "off") ? zfunc(depth) : GL_ALWAYS); p_glDepthMask(depthwrite ? GL_TRUE : GL_FALSE); }
    if (have_stencil) { p_glEnable(GL_STENCIL_TEST); p_glStencilFunc(zfunc(st[0]), atoi(st[1]), (GLuint)strtoul(st[2], 0, 0));
        p_glStencilMask((GLuint)strtoul(st[3], 0, 0)); p_glStencilOp(sop(st[4]), sop(st[5]), sop(st[6])); }
    if (strcmp(cull, "none")) { p_glEnable(GL_CULL_FACE); p_glCullFace(!strcmp(cull, "back") ? GL_BACK : !strcmp(cull, "front") ? GL_FRONT : GL_FRONT_AND_BACK); }
    p_glFrontFace(!strcmp(ff, "cw") ? GL_CW : GL_CCW);
    if (scissor[2] >= 0) { p_glEnable(GL_SCISSOR_TEST); p_glScissor(scissor[0], scissor[1], scissor[2], scissor[3]); }
    if (!strcmp(blend, "alpha")) { p_glEnable(GL_BLEND); p_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); }
    else if (!strcmp(blend, "add")) { p_glEnable(GL_BLEND); p_glBlendFunc(GL_ONE, GL_ONE); }
    if (restart) p_glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    GLenum mode = !strcmp(prim, "strip") ? GL_TRIANGLE_STRIP : !strcmp(prim, "fan") ? GL_TRIANGLE_FAN : GL_TRIANGLES;

    for (int fr = 0; fr < frames; fr++) {
        if (scissor[2] >= 0) p_glDisable(GL_SCISSOR_TEST);
        p_glClearColor(clear[0], clear[1], clear[2], clear[3]); p_glClearDepthf(cleardepth); p_glClearStencil(0);
        p_glClear(GL_COLOR_BUFFER_BIT | (have_depth || have_stencil ? GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT : 0));
        if (scissor[2] >= 0) p_glEnable(GL_SCISSOR_TEST);
        for (int k = 0; k < ndraws; k++) {
            if (indexed) p_glDrawElementsInstanced(mode, draws[k].count, GL_UNSIGNED_SHORT, (void *)(intptr_t)(2 * draws[k].first), draws[k].inst);
            else p_glDrawArraysInstanced(mode, draws[k].first, draws[k].count, draws[k].inst);
        }
        p_glFlush();
    }
    p_glFinish();
    static unsigned char px[4096 * 4096 * 4];
    if (samples == 0 && W * H <= 4096 * 4096) {
        p_glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
        FILE *o = fopen(argv[2], "wb"); if (!o) fail("out"); fwrite(px, 4, (size_t)W * H, o); fclose(o);
    }
    GLenum e = p_glGetError(); if (e != GL_NO_ERROR) { fprintf(stderr, "glError 0x%x\n", e); fail("gl error"); }
    printf("{\"ok\":true,\"W\":%d,\"H\":%d,\"verts\":%d,\"draws\":%d}\n", W, H, nverts, ndraws);
    return 0;
}
