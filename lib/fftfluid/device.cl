#include <pyopencl-complex.h>

size_t here_x(void)
{
    return get_global_id(0);
}

size_t here_y(void) 
{
    return get_global_id(1);
}

size_t width(void) 
{
    return get_global_size(0);
}

size_t height(void)
{
    return get_global_size(1);
}

size_t index(size_t x, size_t y)
{
    return y + x * height();
}

size_t global_index(void)
{
    return index(here_x(), here_y());
}

__kernel void add_force(
    __global const float *fx,
    __global const float *fy,
    __global float *vx,
    __global float *vy,
    float dt)
{
    size_t i = global_index();
    vx[i] += fx[i] * dt;
    vy[i] += fy[i] * dt;
}

float bilerp(
    __global const float *f, 
    float gx, float gy, 
    int i1, int j1, int i2, int j2)
{
    float q11 = f[index(i1, j1)];
    float q12 = f[index(i1, j2)];
    float q21 = f[index(i2, j1)];
    float q22 = f[index(i2, j2)];
    float f1j = q11 * (1.0f - gy) + q12 * gy;
    float f2j = q21 * (1.0f - gy) + q22 * gy;
    float fij = f1j * (1.0f - gx) + f2j * gx;
    return fij;
}

__kernel void advect(
    __global const float *vx,
    __global const float *vy,
    __global cfloat_t *vx_next,
    __global cfloat_t *vy_next,
    float dt)
{
    // Solve advection by semi-Lagrangian method,
    int w = width();
    int h = height();
    int i = here_x();
    int j = here_y();

    float x = 1.0/w * i;
    float y = 1.0/h * j;
    float prev_x = w * (x - vx[index(i, j)] * dt);
    float prev_y = h * (y - vy[index(i, j)] * dt);

    int i1 = floor(prev_x);
    int j1 = floor(prev_y);
    float gx = prev_x - (float) i1;
    float gy = prev_y - (float) j1;
    i1 = (w + (i1 % w)) % w;
    j1 = (h + (j1 % h)) % h;
    int i2 = (i1 + 1) % w;
    int j2 = (j1 + 1) % h;
    float rx = bilerp(vx, gx, gy, i1, j1, i2, j2);
    float ry = bilerp(vy, gx, gy, i1, j1, i2, j2);

    vx_next[index(i, j)] = cfloat_new(rx, 0.0f);
    vy_next[index(i, j)] = cfloat_new(ry, 0.0f);
}

float wave_number(int i, int max)
{
    if (i <= max / 2)
        return i;
    return i - max;
}

__kernel void diffuse_project(
    __global cfloat_t *vx,
    __global cfloat_t *vy,
    __global cfloat_t *p,
    float v,
    float dt)
{
    // Solve diffusion and mass conservation
    float x = here_x();
    float y = here_y();
    float kx = wave_number(x, width());
    float ky = wave_number(y, height());
    float k2 = kx*kx + ky*ky;

    if (k2 != 0.0f) {
        float diff = 1.0f / (1.0f + v * dt * k2); //exp(-k2 * dt * v);
        size_t i = index(x, y);
        cfloat_t vx1 = vx[i];
        cfloat_t vy1 = vy[i];
        float fxx = (1.0f - kx * kx / k2);
        float fxy = (kx * ky / k2);
        float fyx = (-ky * kx / k2);
        float fyy = (1.0f - ky * ky / k2);
        cfloat_t vx2 = cfloat_rmul(diff, cfloat_sub(cfloat_mulr(vx1, fxx), cfloat_mulr(vy1, fxy)));
        cfloat_t vy2 = cfloat_rmul(diff, cfloat_add(cfloat_mulr(vx1, fyx), cfloat_mulr(vy1, fyy)));
        vx[i] = vx2;
        vy[i] = vy2;
    }
}

/*
__kernel void diffuse_project(
    __global cfloat_t *vx,
    __global cfloat_t *vy,
    __global cfloat_t *p,
    float v,
    float dt)
{
    // Solve diffusion and mass conservation
    float x = here_x();
    float y = here_y();
    float kx = wave_number(x, width());
    float ky = wave_number(y, height());
    float k2 = kx*kx + ky*ky;

    if (k2 != 0.0f) {
        float fxx = (1.0f - kx * kx / k2);
        float fxy = (kx * ky / k2);
        float fyx = (-ky * kx / k2);
        float fyy = (1.0f - ky * ky / k2);

        size_t i = index(x, y);
        cfloat_t vx1 = vx[i];
        cfloat_t vy1 = vy[i];
        cfloat_t proj_vx = cfloat_sub(cfloat_mulr(vx1, fxx), cfloat_mulr(vy1, fxy));
        cfloat_t proj_vy = cfloat_add(cfloat_mulr(vx1, fyx), cfloat_mulr(vy1, fyy));

        float diff = 1.0f / (1.0f + v * dt * k2); //exp(-k2 * dt * v);
        vx[i] = cfloat_rmul(diff, proj_vx);
        vy[i] = cfloat_rmul(diff, proj_vy);
    }
}
*/


__kernel void to_real(
    __global const cfloat_t *cvx,
    __global const cfloat_t *cvy,
    __global float *vx,
    __global float *vy)
{
    size_t i = global_index();
    vx[i] = cfloat_real(cvx[i]);
    vy[i] = cfloat_real(cvy[i]);
}
