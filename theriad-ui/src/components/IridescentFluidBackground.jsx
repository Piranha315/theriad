/**
 * IridescentFluidBackground
 * -----------------------------------------------------------------------------
 * Subtle, full-screen, interactive metallic/iridescent background.
 *
 * - Cursor movement produces gentle fluid displacement (a smoothed "wake"
 *   that follows the pointer). Weaker than a click ripple by design.
 * - Clicking spawns a puddle-like ripple: a sharp leading ring plus a
 *   softer, slower trailing wave, both decaying smoothly.
 * - The color field itself is driven by 3 layered "curl" flow fields
 *   (different direction/scale/speed each) instead of a single shifting
 *   gradient, plus a slow overall directional drift and a fine
 *   high-frequency detail term — this is what gives the liquid-metal
 *   swirl rather than a flat moving blob. The field is then treated as
 *   thin-film thickness: the palette cycles through it several times
 *   (VISUAL_TUNING.stripeCount, seamless triangle-wave ping-pong) and a
 *   second-order domain warp bends the resulting fringes into irregular
 *   curls — multiple abstract flowy stripes, like an oil slick or a
 *   curved, crumpled sheet of metal, rather than one diagonal band. A
 *   cheap fake-normal/specular pass adds defined metallic highlight
 *   bands (following the stripe creases) and thin pearlescent streaks
 *   on top, still low-contrast/no hard white lines.
 * - Rendered with a single fullscreen-quad + fragment shader via Three.js —
 *   no fluid simulation, no physics library; cheap enough for continuous
 *   rendering on modest hardware.
 * - Fixed, full-viewport, `pointer-events: none`, negative z-index: it never
 *   intercepts clicks and never affects layout. Pointer/click listeners are
 *   attached to `window`, not the canvas, so the effect still reacts no
 *   matter what element the user actually clicked.
 * - Dark-mode only (matches the existing static iridescent treatment in
 *   index.css): invisible in light mode via a plain CSS opacity rule, and
 *   the render loop skips the actual draw call while light mode is active
 *   to avoid burning GPU/battery for an effect nobody sees.
 * - The render loop also pauses while `document.hidden` (backgrounded tab)
 *   instead of continuing to tick in the background.
 * - Render resolution is capped by device pixel ratio, with a lower cap on
 *   mobile/touch viewports (see VISUAL_TUNING.maxDprMobile) to keep frame
 *   cost reasonable on weaker GPUs.
 * - `prefers-reduced-motion: reduce` disables the WebGL/animation path
 *   entirely; the canvas stays blank and the existing static dark-mode
 *   background (blurred PNG + scrim, in index.css) is the fallback. Same
 *   fallback applies if WebGL/Three.js fails to initialize for any reason
 *   (old GPU, disabled WebGL, etc.) — this component must never throw or
 *   block app startup.
 * - No React state is ever updated per animation frame — all per-frame
 *   mutation happens on plain refs/closures/Three.js objects, so the
 *   60fps loop never triggers a React re-render.
 *
 * Mount ONCE, near the root of the app shell (see App.jsx), before all
 * other content so it paints behind everything by DOM order + z-index.
 */
import { useEffect, useRef } from 'react';
// Dynamically imported inside the effect (not statically here) so Three.js
// — a sizable dependency — is code-split into its own chunk and only
// fetched for visitors who'll actually see the effect (motion allowed,
// WebGL present), instead of bloating everyone's main bundle.

const MAX_RIPPLES = 6;
const RIPPLE_LIFETIME_SECONDS = 4.2; // must match the cutoff used in the fragment shader

/**
 * Single source of truth for the "feel" of the effect. Some values are fed
 * directly into shader uniforms (flowStrength, the highlight/streak
 * strengths); others use different units inside the shader's normalized
 * coordinate space and instead *informed* the hand-tuned constants used
 * below (documented inline at each use site) — a couple of these numbers
 * (e.g. rippleClickStrength) aren't dimensionally meaningful as a literal
 * shader displacement, so they're kept here as an intensity reference
 * rather than passed through as-is.
 */
const VISUAL_TUNING = {
  flowStrength: 0.085,
  flowComplexity: 2.5,
  primaryFlowSpeed: 0.015,
  secondaryFlowSpeed: 0.085,
  tertiaryFlowSpeed: 0.07,

  // How many times the teal->blue->violet palette cycles across the field.
  // This is what turns the old single broad band into multiple thin-film
  // interference stripes (oil-slick look). The palette is ping-ponged via a
  // triangle wave so cycles join seamlessly — no hard color seams.
  stripeCount: 3.0,
  // Strength of the second-order domain warp applied just before the color
  // field is sampled. This is what bends the stripes into irregular, flowy
  // curls (crumpled-sheet-metal look) instead of parallel diagonal bars.
  stripeWarpStrength: 0.19,

  rippleMoveStrength: 0.16,
  rippleClickStrength: 0.1,
  rippleDecay: 0.965,
  rippleRadius: 0.18,
  rippleRingSharpness: 0.022,

  metallicHighlightStrength: 0.34,
  pearlescentBandStrength: 0.22,
  detailStrength: 0.4,

  maxDprDesktop: 1.5,
  maxDprMobile: 1.15,
};

// Derived shader inputs. See the VISUAL_TUNING comment above for why these
// aren't all a straight pass-through of the tuning numbers.
const POINTER_RIPPLE_AMPLITUDE = 0.06;   // was 0.045 — softer than click, ~+33%
const CLICK_RIPPLE_PRIMARY_AMPLITUDE = 0.06;  // was 0.03 — leading ring, +100%
const CLICK_RIPPLE_TRAILING_AMPLITUDE = 0.028; // new secondary/trailing wave
const CLICK_RIPPLE_RADIUS_FALLOFF = 2.1; // was 2.6 — slightly wider radius
const CLICK_RIPPLE_DECAY_RATE = 1.15;    // was 1.5 — slightly slower decay

const VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Layered "liquid metal" field: 3 curl-noise flow layers (different
// direction/scale/speed) plus a slow overall drift and a fine detail term
// drive the color field; click/pointer ripples add local displacement on
// top; a cheap fake-normal specular pass adds defined highlight bands.
// Colors stay restrained (brand teal -> blue -> violet), and the whole
// result is emitted at low alpha so it stays subtle under the app's dark
// overlay — more internal structure, not more brightness.
const FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform float uTime;
  uniform vec2 uResolution;
  uniform vec2 uMouse;
  uniform vec2 uRippleOrigin[${MAX_RIPPLES}];
  uniform float uRippleTime[${MAX_RIPPLES}];
  uniform float uOpacity;
  uniform float uFlowStrength;
  uniform float uMetallicHighlightStrength;
  uniform float uPearlescentBandStrength;
  uniform float uDetailStrength;
  uniform float uStripeCount;
  uniform float uStripeWarpStrength;

  varying vec2 vUv;

  const vec2 FLOW_DIR_1 = vec2(0.8767, 0.4813);   // normalize(vec2(1.0, 0.55))
  const vec2 FLOW_DIR_2 = vec2(-0.5280, 0.8492);  // normalize(vec2(-0.65, 1.05))
  const vec2 FLOW_DIR_3 = vec2(0.9433, -0.3319);  // normalize(vec2(1.15, -0.4))
  const float FLOW_SCALE_1 = 1.2;
  const float FLOW_SCALE_2 = 2.7;
  const float FLOW_SCALE_3 = 5.5;
  const float FLOW_SPEED_1 = 0.045;
  const float FLOW_SPEED_2 = 0.085;
  const float FLOW_SPEED_3 = 0.13;
  const float FLOW_WEIGHT_1 = 0.55;
  const float FLOW_WEIGHT_2 = 0.32;
  const float FLOW_WEIGHT_3 = 0.16;

  vec3 palette(float t) {
    vec3 teal   = vec3(0.16, 0.1, 0.58);
    vec3 blue   = vec3(0.2, 0.4, 0.82);
    vec3 violet = vec3(0.40, 0.6, 0.95);
    vec3 c = mix(teal, blue, smoothstep(0.04, 0.5, t));
    c = mix(c, violet, smoothstep(0.5, 0.96, t));
    return c;
  }

  // Analytic gradient of a single moving sine wave: cheap (one cos, no
  // finite differences). grad also doubles as the input to the fake
  // surface normal used for the metallic highlight below.
  void flowLayer(vec2 p, vec2 dir, float scale, float speed, float t, out vec2 grad) {
    vec2 k = dir * scale;
    float phase = dot(k, p) + t * speed;
    grad = k * cos(phase);
  }

  void main() {
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2 p = (vUv - 0.5);
    p.x *= aspect;

    // Slow large-scale drift/current, independent of each flow layer's own
    // speed — gives the whole field a gentle directional pan, like a
    // current under the surface rather than a stationary pattern.
    vec2 drift = vec2(uTime * 0.010, uTime * 0.006);
    vec2 fp = p + drift;

    vec2 g1, g2, g3;
    flowLayer(fp, FLOW_DIR_1, FLOW_SCALE_1, FLOW_SPEED_1, uTime, g1);
    flowLayer(fp, FLOW_DIR_2, FLOW_SCALE_2, FLOW_SPEED_2, uTime, g2);
    flowLayer(fp, FLOW_DIR_3, FLOW_SCALE_3, FLOW_SPEED_3, uTime, g3);

    // Divergence-free ("curl") field: rotate each layer's gradient 90
    // degrees. This is what produces swirling, liquid-like advection
    // instead of a plain shifting gradient.
    vec2 flow =
      (vec2(g1.y, -g1.x) * FLOW_WEIGHT_1 +
       vec2(g2.y, -g2.x) * FLOW_WEIGHT_2 +
       vec2(g3.y, -g3.x) * FLOW_WEIGHT_3) * uFlowStrength;

    vec2 warp = flow;

    // Continuous fluid "wake" that gently trails the cursor — kept softer
    // than a click ripple.
    vec2 mouseP = (uMouse - 0.5);
    mouseP.x *= aspect;
    float mouseDist = length(p - mouseP);
    vec2 mouseDir = (p - mouseP) / max(mouseDist, 0.0001);
    float mouseFalloff = exp(-mouseDist * 2.2);
    warp += mouseDir * mouseFalloff * ${POINTER_RIPPLE_AMPLITUDE.toFixed(4)}
            * sin(uTime * 0.6 - mouseDist * 6.0);

    // Click ripples: a sharp leading ring plus a softer, slower trailing
    // wave that fades in just after it — reads as a puddle, not a single
    // flat pulse. Both decay smoothly (exponential in age and distance).
    for (int i = 0; i < ${MAX_RIPPLES}; i++) {
      float age = uRippleTime[i];
      if (age < 0.0 || age > ${RIPPLE_LIFETIME_SECONDS.toFixed(2)}) continue;
      vec2 o = (uRippleOrigin[i] - 0.5);
      o.x *= aspect;
      float d = length(p - o);
      vec2 dir = (p - o) / max(d, 0.0001);

      float radiusFalloff = exp(-d * ${CLICK_RIPPLE_RADIUS_FALLOFF.toFixed(2)});
      float ageDecay = exp(-age * ${CLICK_RIPPLE_DECAY_RATE.toFixed(2)});

      float primaryRing = sin(d * 30.0 - age * 9.0) * ageDecay * radiusFalloff;
      float trailingWave = sin(d * 13.0 - age * 5.0 - 1.2)
                            * exp(-age * 0.85) * exp(-d * 1.4)
                            * smoothstep(0.0, 0.6, age);

      warp += dir * (primaryRing * ${CLICK_RIPPLE_PRIMARY_AMPLITUDE.toFixed(4)}
                     + trailingWave * ${CLICK_RIPPLE_TRAILING_AMPLITUDE.toFixed(4)});
    }

    vec2 wp = p + warp;

    // Second-order domain warp: re-sample one extra curl layer AT the
    // already-warped position and displace again. This "warp of a warp" is
    // what bends the stripe pattern below into irregular, flowy curls —
    // like ripples in a crumpled metal sheet — instead of straight bands.
    vec2 g4;
    flowLayer(wp + drift * 0.5, vec2(0.2747, 0.9615), 3.3, 0.05, uTime, g4);
    wp += vec2(g4.y, -g4.x) * uStripeWarpStrength;

    float field = 0.0;
    field += sin(wp.x * 2.2 + uTime * 0.045);
    field += sin(wp.y * 2.6 - uTime * 0.037);
    field += sin((wp.x + wp.y) * 1.5 + uTime * 0.028);
    field /= 3.0;

    // Fine high-frequency detail riding on top of the base field — adds
    // visible internal structure without changing the overall motion.
    field += uDetailStrength * sin(wp.x * 9.0 - wp.y * 7.0 + uTime * 0.22);
    field = clamp(field * 0.5 + 0.5, 0.0, 1.0);

    // --- Thin-film striping (the oil-slick move) -----------------------
    // Treat the field as film thickness and cycle the palette through it
    // uStripeCount times instead of once. A triangle wave ping-pongs the
    // ramp (teal->violet->teal->...) so consecutive cycles join without a
    // seam. The tiny uTime term lets the fringes migrate very slowly.
    float bandT = field * uStripeCount + uTime * 0.012;
    float tri = abs(fract(bandT) * 2.0 - 1.0);
    vec3 color = palette(tri);

    // --- Metallic highlight -------------------------------------------
    // Fake surface normal from two parts: the broad flow gradient (soft
    // overall sheen, as before) plus the analytic gradient of the banded
    // field (sharp sheen riding along each stripe edge). The second term
    // is what makes the stripes read as creases in curved metal rather
    // than flat printed color.
    vec2 gradSum = g1 * FLOW_WEIGHT_1 + g2 * FLOW_WEIGHT_2 + g3 * FLOW_WEIGHT_3;
    // d(field)/d(wp), analytic (cheap: three cos, no finite differences)
    vec2 fieldGrad = 0.5 * vec2(
      (2.2 * cos(wp.x * 2.2 + uTime * 0.045)
        + 1.5 * cos((wp.x + wp.y) * 1.5 + uTime * 0.028)) / 3.0,
      (2.6 * cos(wp.y * 2.6 - uTime * 0.037)
        + 1.5 * cos((wp.x + wp.y) * 1.5 + uTime * 0.028)) / 3.0
    );
    // Height slope of the banded surface: chain rule through sin(2*pi*bandT)
    vec2 bandGrad = cos(6.28318 * bandT) * uStripeCount * fieldGrad;
    vec3 normal = normalize(vec3(-(gradSum * 0.35 + bandGrad * 0.55), 1.0));
    vec3 lightDir = normalize(vec3(0.35, 0.55, 0.72));
    float spec = pow(max(dot(normal, lightDir), 0.0), 10.0);
    color += vec3(0.85, 0.92, 1.0) * spec * uMetallicHighlightStrength;

    // Thin pearlescent streaks: a narrow smoothstep band on a slower,
    // higher-frequency field — chrome-like ribbons, deliberately kept
    // low-contrast (no hard white lines, no neon glow).
    float streakField = sin(wp.x * 6.5 - wp.y * 3.0 + uTime * 0.02);
    float streak = smoothstep(0.93, 1.0, streakField);
    color += palette(1.0 - field) * streak * uPearlescentBandStrength;

    gl_FragColor = vec4(color, uOpacity);
  }
`;

function isMobileViewport() {
  const narrow = typeof window.matchMedia === 'function'
    && window.matchMedia('(max-width: 768px)').matches;
  const touch = (navigator.maxTouchPoints || 0) > 2;
  return narrow || touch;
}

export default function IridescentFluidBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const media = typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;

    // Reduced motion (or no matchMedia support): leave the canvas blank.
    // The static dark-mode background already defined in index.css is the
    // fallback — nothing else to do here. Skips the Three.js import too.
    if (media?.matches) return undefined;

    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    let cancelled = false;
    let cleanup = () => {};

    // Three.js is loaded dynamically (see import comment above) so it never
    // lands in the main bundle for visitors who never reach this branch.
    // React may unmount before this resolves (fast navigation, StrictMode's
    // double-invoke in dev) — `cancelled` guards against setting up on a
    // component that's already gone.
    import('three').then((THREE) => {
      if (cancelled) return;
      cleanup = setupScene(THREE, canvas);
    }).catch(() => {
      // Failed to load/init Three.js — static fallback remains, no crash.
    });

    return () => {
      cancelled = true;
      cleanup();
    };
  }, []);

  return (
    <div className="iridescent-fluid-bg" aria-hidden="true">
      <canvas ref={canvasRef} />
      <div className="iridescent-fluid-bg__overlay" />
    </div>
  );
}

/**
 * Builds the Three.js scene/renderer against an already-mounted <canvas>
 * and wires up the animation loop + listeners. Returns a cleanup function.
 * Broken out from the effect so the dynamic `import('three')` above stays
 * simple (`.then(setupScene)`-shaped) rather than an inline async IIFE.
 */
function setupScene(THREE, canvas) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'low-power',
    });
  } catch {
    // WebGL unavailable/blocked — static fallback remains, app must not crash.
    return () => {};
  }

  const maxDpr = isMobileViewport() ? VISUAL_TUNING.maxDprMobile : VISUAL_TUNING.maxDprDesktop;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxDpr));

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const rippleOrigin = Array.from({ length: MAX_RIPPLES }, () => new THREE.Vector2(0.5, 0.5));
  const rippleTime = new Float32Array(MAX_RIPPLES).fill(-1);
  let rippleCursor = 0;

  const uniforms = {
    uTime: { value: 0 },
    uResolution: { value: new THREE.Vector2(1, 1) },
    uMouse: { value: new THREE.Vector2(0.5, 0.5) },
    uRippleOrigin: { value: rippleOrigin },
    uRippleTime: { value: rippleTime },
    uOpacity: { value: 0.4 },
    uFlowStrength: { value: VISUAL_TUNING.flowStrength },
    uMetallicHighlightStrength: { value: VISUAL_TUNING.metallicHighlightStrength },
    uPearlescentBandStrength: { value: VISUAL_TUNING.pearlescentBandStrength },
    uDetailStrength: { value: VISUAL_TUNING.detailStrength },
    uStripeCount: { value: VISUAL_TUNING.stripeCount },
    uStripeWarpStrength: { value: VISUAL_TUNING.stripeWarpStrength },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });

  const geometry = new THREE.PlaneGeometry(2, 2);
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    uniforms.uResolution.value.set(w, h);
  }
  resize();

  const targetMouse = new THREE.Vector2(0.5, 0.5);
  const smoothedMouse = new THREE.Vector2(0.5, 0.5);

  function onPointerMove(e) {
    targetMouse.set(e.clientX / window.innerWidth, 1 - e.clientY / window.innerHeight);
  }

  function onPointerDown(e) {
    const x = e.clientX / window.innerWidth;
    const y = 1 - e.clientY / window.innerHeight;
    rippleOrigin[rippleCursor].set(x, y);
    rippleTime[rippleCursor] = 0;
    rippleCursor = (rippleCursor + 1) % MAX_RIPPLES;
  }

  // Listen on window (not the canvas): the canvas is pointer-events:none,
  // so this is the only way to react to pointer activity anywhere in the
  // app while guaranteeing the background never intercepts a click.
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('pointerdown', onPointerDown, { passive: true });
  window.addEventListener('resize', resize);

  const clock = new THREE.Clock();
  let rafId = null;
  let disposed = false;

  function step() {
    const dt = Math.min(clock.getDelta(), 0.1);

    // Dark-mode only (see file header) — skip the actual draw call
    // outside dark mode so an invisible effect doesn't still cost GPU.
    if (document.documentElement.dataset.theme !== 'dark') return;

    uniforms.uTime.value = clock.elapsedTime;

    // Frame-rate independent exponential smoothing toward the pointer —
    // this is what makes the displacement feel like a gentle fluid trail
    // rather than a snap-to-cursor jump.
    const smoothing = 1 - Math.pow(0.0015, dt);
    smoothedMouse.lerp(targetMouse, smoothing);
    uniforms.uMouse.value.copy(smoothedMouse);

    for (let i = 0; i < MAX_RIPPLES; i++) {
      if (rippleTime[i] >= 0) {
        rippleTime[i] += dt;
        if (rippleTime[i] > RIPPLE_LIFETIME_SECONDS) rippleTime[i] = -1;
      }
    }

    renderer.render(scene, camera);
  }

  function tick() {
    if (disposed) return;
    rafId = requestAnimationFrame(tick);
    step();
  }

  // Pause entirely while the tab is backgrounded (document.hidden) instead
  // of continuing to tick — resumes as soon as it's visible again. clock
  // delta is naturally large across the gap; the dt clamp above (0.1s)
  // keeps that from producing a visible jump on resume.
  function onVisibilityChange() {
    if (document.hidden) {
      if (rafId != null) cancelAnimationFrame(rafId);
      rafId = null;
    } else if (rafId == null && !disposed) {
      tick();
    }
  }
  document.addEventListener('visibilitychange', onVisibilityChange);

  if (!document.hidden) tick();

  return () => {
    disposed = true;
    if (rafId != null) cancelAnimationFrame(rafId);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('resize', resize);
    geometry.dispose();
    material.dispose();
    renderer.dispose();
  };
}
