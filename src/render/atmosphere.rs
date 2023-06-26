#![allow(non_upper_case_globals)]

struct Ray {
    origin: glm::TVec3<f32>,
    direction: glm::TVec3<f32>
}

struct Sphere {
    origin: glm::TVec3<f32>,
    radius: f32
}


// scale height (m)
// thickness of the atmosphere if its density were uniform
const hR: f32 = 7994.0; // Rayleigh
const hM: f32 = 1200.0; // Mie

fn rayleigh_phase_func(mu: f32) -> f32
{
    return
            3.0 * (1.0 + mu * mu)
    / //------------------------
           (16.0 * glm::pi::<f32>());
}

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
const g: f32 = 0.76;
fn henyey_greenstein_phase_func(mu: f32) -> f32
{
    return
                        (1.0 - g * g)
    / //---------------------------------------------
           ((4.0 * glm::pi::<f32>()) * f32::powf(1.0 + g * g - 2.0 * g * mu, 1.5));
}

// Schlick Phase Function factor
// Pharr and  Humphreys [2004] equivalence to g above
#[allow(unused)]
const k: f32 = 1.55 * g - 0.55 * (g * g * g);
#[allow(unused)]
fn schlick_phase_func(mu: f32) -> f32
{
    return
                    (1.0 - k * k)
    / //-------------------------------------------
           (4.0 * glm::pi::<f32>() * (1.0 + k * mu) * (1.0 + k * mu));
}

fn isect_sphere(ray: &Ray, sphere: &Sphere, t0: &mut f32, t1: &mut f32) -> bool
{
    let rc = sphere.origin - ray.origin;
    let radius2 = sphere.radius * sphere.radius;
    let tca = glm::dot(&rc, &ray.direction);
    let d2 = glm::dot(&rc, &rc) - tca * tca;
    if d2 > radius2 { return false; }
    let thc = f32::sqrt(radius2 - d2);
    *t0 = tca - thc;
    *t1 = tca + thc;

    return true;
}

pub const EARTH_RADIUS: f32 = 6360e3;      // (m)
const ATMOSPHERE_RADIUS: f32 = 6420e3; // (m)

#[allow(unused)]
fn get_primary_ray(
    cam_local_point: &glm::TVec3<f32>,
    cam_origin: &mut glm::TVec3<f32>,
    cam_look_at: &mut glm::TVec3<f32>
) -> Ray {
    let fwd = glm::normalize(&(*cam_look_at - *cam_origin));
    let up = glm::vec3(0.0, 1.0, 0.0);
    let right = glm::cross(&up, &fwd);
    let up = glm::cross(&fwd, &right);

    let r = Ray {
        origin: *cam_origin,
        direction: glm::normalize(&(fwd + up * cam_local_point.y + right * cam_local_point.x))
    };

    return r;
}

fn get_sun_light(
    ray: &Ray,
    optical_depthR: &mut f32,
    optical_depthM: &mut f32,
) -> bool {
    const NUM_SAMPLES: u32 = 8;
    
    let atmosphere = Sphere {
        origin: glm::zero(),
        radius: ATMOSPHERE_RADIUS
    };

    let mut t0 = 0.0;
    let mut t1 = 0.0;
    isect_sphere(&ray, &atmosphere, &mut t0, &mut t1);

    let mut march_pos = 0.0;
    let march_step = t1 / (NUM_SAMPLES as f32);

    for _ in 0..NUM_SAMPLES {
        let s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height = glm::length(&s) - EARTH_RADIUS;
        if height < 0.0 { return false; }

        *optical_depthR += f32::exp(-height / hR) * march_step;
        *optical_depthM += f32::exp(-height / hM) * march_step;

        march_pos += march_step;
    }

    return true;
}

pub fn gather_atmosphere_irradiance(origin: &glm::TVec3<f32>, direction: &glm::TVec3<f32>, sun_dir: &glm::TVec3<f32>, sun_irradiance: &glm::TVec3<f32>) -> glm::TVec3<f32> {
    const NUM_SAMPLES: u32 = 16;
    let atmosphere = Sphere {
        origin: glm::zero(),
        radius: ATMOSPHERE_RADIUS
    };

    let ray = Ray {
        origin: *origin,
        direction: *direction
    };

    // scattering coefficients at sea level (m)
    let betaR: glm::TVec3<f32> = glm::vec3(5.5e-6, 13.0e-6, 22.4e-6);     //Rayleigh
    let betaM: glm::TVec3<f32> = glm::vec3(21e-6, 21e-6, 21e-6);          //Mie

    // "pierce" the atmosphere with the viewing ray
    let mut t0 = 0.0;
    let mut t1 = 0.0;
    let did_isect = isect_sphere(&ray, &atmosphere, &mut t0, &mut t1);
    if !did_isect {
        return glm::vec3(0.6, 0.6, 0.6);
    }

    // cosine of angle between view and light directions
    let mu = glm::dot(&ray.direction, &sun_dir);

    // Rayleigh and Mie phase functions
    // A black box indicating how light is interacting with the material
    // Similar to BRDF except
    // * it usually considers a single angle
    //   (the phase angle between 2 directions)
    // * integrates to 1 over the entire sphere of directions
    let phaseR = rayleigh_phase_func(mu);
    let phaseM = henyey_greenstein_phase_func(mu);

    // optical depth (or "average density")
    // represents the accumulated extinction coefficients
    // along the path, multiplied by the length of that path
    let mut optical_depthR = 0.0;
    let mut optical_depthM = 0.0;

    let mut sumR: glm::TVec3<f32> = glm::zero();
    let mut sumM: glm::TVec3<f32> = glm::zero();
    let mut march_pos = 0.0;
    let march_step = t1 / (NUM_SAMPLES as f32);

    for _ in 0..NUM_SAMPLES {
        let s = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height = glm::length(&s) - EARTH_RADIUS;

        // integrate the height scale
        let hr = f32::exp(-height / hR) * march_step;
        let hm = f32::exp(-height / hM) * march_step;
        optical_depthR += hr;
        optical_depthM += hm;

        // gather the sunlight
        let light_ray = Ray {
            origin: s,
            direction: *sun_dir
        };
        let mut optical_depth_lightR = 0.0;
        let mut optical_depth_lightM = 0.0;
        let overground = get_sun_light(&light_ray, &mut optical_depth_lightR, &mut optical_depth_lightM);

        if overground {
            let tau =
                betaR * (optical_depthR + optical_depth_lightR) +
                betaM * 1.1 * (optical_depthM + optical_depth_lightM);
            let attenuation = glm::exp(&-tau);

            sumR += hr * attenuation;
            sumM += hm * attenuation;
        }

        march_pos += march_step;
    }

    let r_term = phaseR * glm::matrix_comp_mult(&sumR, &betaR);
    let m_term = phaseM * glm::matrix_comp_mult(&sumM, &betaM);
    return glm::matrix_comp_mult(sun_irradiance, &(r_term + m_term));
}