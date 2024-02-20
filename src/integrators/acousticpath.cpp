#include <mutex>
#include <tuple>

#include <mitsuba/core/fwd.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <nanothread/nanothread.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AcousticPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_samples_per_pass, m_max_depth, m_rr_depth,
                   m_hide_emitters, m_render_timer, m_stop)
    MI_IMPORT_TYPES(Scene, Sensor, Film, ImageBlock, Medium, Sampler, BSDFPtr)

    AcousticPathIntegrator(const Properties &props) : Base(props) {
        m_max_time    = props.get<float>("max_time", 1.f);
        m_speed_of_sound = props.get<float>("speed_of_sound", 343.f);
        if (m_max_time <= 0.f || m_speed_of_sound <= 0.f)
            Throw("\"max_time\" and \"speed_of_sound\" must be set to a value greater than zero!");

        m_skip_direct       = props.get<bool>("skip_direct", false);

        int max_depth = props.get<int>("max_depth", -1);
        if (max_depth < 0 && max_depth != -1)
            Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");

        m_max_depth = (uint32_t) max_depth; // This maps -1 to 2^32-1 bounces

        // Depth to begin using russian roulette
        int rr_depth = props.get<int>("rr_depth", m_max_depth);
        if (rr_depth <= 0)
            Throw("\"rr_depth\" must be set to a value greater than zero!");

        m_rr_depth = (uint32_t) rr_depth;

        m_verbose = props.get<bool>("verbose", false);
    }

    TensorXf render(Scene *scene,
                    Sensor *sensor,
                    uint32_t seed,
                    uint32_t spp,
                    bool develop,
                    bool evaluate) override {
        ScopedPhase sp(ProfilerPhase::Render);
        m_stop = false;

        Film *film = sensor->film();
        ScalarVector2u film_size = film->crop_size();

        // Potentially adjust the number of samples per pixel if spp != 0
        Sampler *sampler = sensor->sampler();
        if (spp)
            sampler->set_sample_count(spp);
        spp = sampler->sample_count();

        uint32_t spp_per_pass = (m_samples_per_pass == (uint32_t) -1)
                                        ? spp
                                        : std::min(m_samples_per_pass, spp);

        if ((spp % spp_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of spp_per_pass (%d).",
                  spp, spp_per_pass);

        uint32_t n_passes = spp / spp_per_pass;

        // Determine output channels and prepare the film with this information
        size_t n_channels = film->prepare({});
        Log(Debug, "Rendering with %u channel%s", n_channels, n_channels == 1 ? "" : "s");

        if (has_flag(sensor->film()->flags(), FilmFlags::Spectral))
            Log(Info, "Spectral film detected, sampling continuous wavelengths.");
        else
            Log(Info, "No spectral film detected, iterating through wavelength bins.");


        m_render_timer.reset();

        TensorXf result;
        if constexpr (!dr::is_jit_v<Float>) {
            // Render on the CPU using a spiral pattern
            uint32_t n_threads = (uint32_t) Thread::thread_count();

            Log(Info, "Starting render job (%ux%u, %u sample%s,%s %u thread%s)",
                film_size.x(), film_size.y(), spp, spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(" %u passes,", n_passes) : "", n_threads,
                n_threads == 1 ? "" : "s");

            std::mutex mutex;
            ref<ProgressReporter> progress;
            Logger* logger = mitsuba::Thread::thread()->logger();
            if (logger && Info >= logger->log_level())
                progress = new ProgressReporter("Rendering");

            // Total number of blocks to be handled, including multiple passes.
            uint32_t total_blocks = film_size.x() * n_passes,
                     blocks_done = 0;
            Log(Debug, "Total blocks: %u", total_blocks);

            // Avoid overlaps in RNG seeding RNG when a seed is manually specified
            seed *= dr::prod(film_size);

            ThreadEnvironment env;
            dr::parallel_for(
                dr::blocked_range<uint32_t>(0, total_blocks, 1),
                [&](const dr::blocked_range<uint32_t> &range) {
                    ScopedSetThreadEnvironment set_env(env);
                    // Fork a non-overlapping sampler for the current worker
                    ref<Sampler> sampler = sensor->sampler()->fork();
                    ref<ImageBlock> block = film->create_block(ScalarVector2u(1, film_size.y()));
                    std::unique_ptr<Float[]> aovs(new Float[n_channels]);

                    for (uint32_t i = range.begin(); i != range.end(); ++i) {
                        sampler->seed(seed * i);

                        UInt32 band_id(i / n_passes);
                        Log(Debug, "Rendering block %u of %u (band %u)", i+1, total_blocks, band_id);
                        block->set_offset(ScalarPoint2u(0, band_id));

                        if constexpr (dr::is_array_v<Float>) {
                            Throw("Not implemented for JIT arrays.");
                        } else {
                            block->clear();

                            for (uint32_t j = 0; j < spp_per_pass; ++j) {
                                Log(Debug, "Rendering sample %u of %u", j+1, spp_per_pass);
                                render_sample(scene, sensor, film, sampler, block, aovs.get(), band_id);
                                sampler->advance();
                            }
                        }

                        film->put_block(block);

                        /* Critical section: update progress bar */
                        if (progress) {
                            std::lock_guard<std::mutex> lock(mutex);
                            blocks_done++;
                            progress->update(blocks_done / (float) total_blocks);
                        }
                    }
                }
            );

            if (develop)
                result = film->develop();
         } else {
            //                               wav_bins      * samples per pixel
            size_t wavefront_size = (size_t) film_size.x() * (size_t) spp_per_pass,
                   wavefront_size_limit = 0xffffffffu;

            if (wavefront_size > wavefront_size_limit) {
                spp_per_pass /=
                    (uint32_t)((wavefront_size + wavefront_size_limit - 1) /
                               wavefront_size_limit);
                n_passes       = spp / spp_per_pass;
                wavefront_size = (size_t) film_size.x() * (size_t) spp_per_pass;

                Log(Warn,
                    "The requested rendering task involves %zu Monte Carlo "
                    "samples, which exceeds the upper limit of 2^32 = 4294967296 "
                    "for this variant. Mitsuba will instead split the rendering "
                    "task into %zu smaller passes to avoid exceeding the limits.",
                    wavefront_size, n_passes);
            }

            dr::sync_thread(); // Separate from scene initialization (for timings)

            Log(Info, "Starting render job (%ux%u, %u sample%s%s)",
                film_size.x(), film_size.y(), spp, spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(", %u passes", n_passes) : "");

            if (n_passes > 1 && !evaluate) {
                Log(Warn, "render(): forcing 'evaluate=true' since multi-pass "
                          "rendering was requested.");
                evaluate = true;
            }

            // Inform the sampler about the passes (needed in vectorized modes)
            sampler->set_samples_per_wavefront(spp_per_pass);

            // Seed the underlying random number generators, if applicable
            sampler->seed(seed, (uint32_t) wavefront_size);

            // Allocate a large image block that will receive the entire rendering
            ref<ImageBlock> block = film->create_block();
            block->set_offset(film->crop_offset());

            UInt32 idx = dr::arange<UInt32>((uint32_t) wavefront_size);

            // Try to avoid a division by an unknown constant if we can help it
            uint32_t log_spp_per_pass = dr::log2i(spp_per_pass);
            if ((1u << log_spp_per_pass) == spp_per_pass)
                idx >>= dr::opaque<UInt32>(log_spp_per_pass);
            else
                idx /= dr::opaque<UInt32>(spp_per_pass);

            Vector2u pos(idx, 0 * idx);

            Timer timer;
            std::unique_ptr<Float[]> aovs(new Float[n_channels]);

            for (size_t i = 0; i < n_passes; i++) {
                render_sample(scene, sensor, film, sampler, block, aovs.get(), pos);

                if (n_passes > 1) {
                    sampler->advance();
                    sampler->schedule_state();
                    dr::eval(block->tensor());
                }
            }

            film->put_block(block);

            if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                jit_flag(JitFlag::LoopRecord)) {
                Log(Info, "Computation graph recorded. (took %s)",
                    util::time_string((float) timer.reset(), true));
            }

            if (develop) {
                result = film->develop();
                dr::schedule(result);
            } else {
                film->schedule_storage();
            }

            if (evaluate) {
                dr::eval();

                if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                    jit_flag(JitFlag::LoopRecord)) {
                    Log(Info, "Code generation finished. (took %s)",
                        util::time_string((float) timer.value(), true));

                    /* Separate computation graph recording from the actual
                       rendering time in single-pass mode */
                    m_render_timer.reset();
                }

                dr::sync_thread();
            }
        }

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return result;
    }

    /// default function signature for proper inheritance
    std::pair<Spectrum, Bool> sample(const Scene *,
                                     Sampler *,
                                     const RayDifferential3f &,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool) const override {
        NotImplementedError("AcousticPathIntegrator::sample default arguments");
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const Film *film,
                                     const Vector2f &pos,
                                     const RayDifferential3f &ray_,
                                     ImageBlock *block,
                                     Float *aovs /* this stores the values that are put into the ImageBlock, see film::prepare_sample() */,
                                     Bool active) const {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                      = Ray3f(ray_);
        Spectrum throughput            = 1.f;
        Float eta                      = 1.f;
        UInt32 depth                   = 0;
        Float distance                 = 0.f;
        const ScalarFloat max_distance = m_max_time * m_speed_of_sound;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray                 = !m_hide_emitters && dr::neq(scene->environment(), nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si          = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf    = m_skip_direct ? 0.f : 1.f;
        Bool          prev_bsdf_delta  = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.

           The first argument identifies the loop by name, which is helpful for
           debugging. The subsequent list registers all variables that encode
           the loop state variables. This is crucial: omitting a variable may
           lead to undefined behavior. */
        // TODO: loop should keep track of imageblock data
        dr::Loop<Bool> loop("AcousticPath", sampler, ray, throughput, eta, depth, distance,
                            valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta, active);

        /* Inform the loop about the maximum number of loop iterations.
           This accelerates wavefront-style rendering by avoiding costly
           synchronization points that check the 'active' flag. */
        loop.set_max_iterations(m_max_depth);

        while (loop(active)) {
            /* dr::Loop implicitly masks all code in the loop using the 'active'
               flag, so there is no need to pass it to every function */
            Log(Debug, "Tracing ray with origin %s, wavelength %f", ray.o, ray.wavelengths);

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ dr::eq(depth, 0u));

            distance += si.t;
            Log(Debug, "Intersection found with distance %f m.", si.t);

            // ---------------------- Direct emission ----------------------

            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            Mask hit_emitter = dr::neq(si.emitter(scene), nullptr);
            Log(Debug, "Hit emitter? %s", hit_emitter);
            if (dr::any_or<true>(hit_emitter)) {
                DirectionSample3f ds(scene, si, prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                          !prev_bsdf_delta);

                /* Compute MIS weight for emitter sample from previous bounce.
                   If em_pdf = 0, then mis_bsdf = 1. This is the case in the first iteration.*/
                Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                Float time_frac = (distance / max_distance) * block->size().y();
                Float data[2] = { (throughput * ds.emitter->eval(si, prev_bsdf_pdf > 0.f) * mis_bsdf).x(), Float(1.f) };
                Float band_id = ray.wavelengths[0]-1;
                if (Spectrum::Size > 1) {
                    Throw("AcousticPathIntegrator does not support multi-wavelength rendering");
                }
                block->put({ band_id, time_frac }, data, hit_emitter && data[0] > 0.f);
            }

            // Continue tracing the path at this point?
            Bool active_next = (depth + 1 < m_max_depth)
                && si.is_valid() && distance <= max_distance;
            Log(Debug, "Continue tracing the path at this point? %s", active_next);
            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            BSDFPtr bsdf = si.bsdf(ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);
            Log(Debug, "Perform Emitter sampling? %s", active_em);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= dr::neq(ds.pdf, 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------

            Log(Debug, "calculating Emitter sampling contribution at wavelength %f",si.wavelengths);
            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
                Log(Debug, "bsdf_val: %s", bsdf_val);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                Log(Debug, "mis_em: %f", mis_em);
                Float time_frac = ((distance + ds.dist) / max_distance) * block->size().y();
                Float data[2] = { (throughput * bsdf_val * em_weight * mis_em).x(), Float(1.f) };
                active_em &= data[0] > 0.f;
                Log(Debug, "time_frac: %f, data: %f, active_em: %s",
                time_frac, data, active_em);
                // TODO: move the put block into render_block to enable spectral post processing?
                // TODO: need to call spectape->prepare_sample to distribute the contribution to the correct channels
                Float band_id = ray.wavelengths[0]-1;
                if (Spectrum::Size > 1) {
                    Throw("AcousticPathIntegrator does not support multi-wavelength rendering");
                }
                block->put({ pos.x(), time_frac }, data, active_em);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ray)) {
                ray = dr::detach<true>(ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight;
            eta *= bsdf_sample.eta;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            prev_si = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            Mask rr_active = depth >= m_rr_depth,
                 rr_continue = sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            active = active_next && (!rr_active || rr_continue) &&
                     dr::neq(throughput_max, 0.f) &&
                     distance <= max_distance;
        }

        if (m_verbose) {
            std::cout << "max: "  << dr::max(depth)
                      << " avg: " << dr::mean(depth)
                      << " var: " << dr::mean(dr::sqr(depth - dr::mean(depth))) << std::endl;
        }

        return {
            /* spec  = */ dr::select(valid_ray, throughput, 0.f),
            /* valid = */ valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AcousticPathIntegrator["
            <<  "\n  stop = "            << m_stop
            << ",\n  max_depth = "       << m_max_depth
            << ",\n  rr_depth = "       << m_rr_depth
            <<  "\n]";
        return oss.str();
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return dr::select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS()

protected:

    void render_sample(const Scene *scene,
                       const Sensor *sensor,
                       const Film *film,
                       Sampler *sampler,
                       ImageBlock *block,
                       Float *aovs, /* just passed through towards sample(), putting data into the block needs to happen inside sample() to avoid storing copies of entire histograms. */
                       const Vector2f &pos,
                       Mask active = true) const {

        Point2f aperture_sample(.5f);
        if (sensor->needs_aperture_sample())
            aperture_sample = sampler->next_2d(active);

        ScalarVector2f scale  = 1.f / ScalarVector2f(sensor->film()->crop_size());
        Vector2f adjusted_pos = pos * scale;

        // check if the film is a tape or spectape. In the future, this could be changed by implementing two acoustic variants:
        // acoustic, with "spectrum": "Color<Float, 1>" and acoustic_spectral, with "spectrum": "Spectrum<Float, 1>"
        Float wavelength_sample = 0.f;
        if (has_flag(sensor->film()->flags(), FilmFlags::Spectral)) {
            wavelength_sample = sampler->next_1d(active);
        }
        else {
            wavelength_sample = pos.x() + 1.f;

        }
        Log(Debug, "Wavelength sample: %f", wavelength_sample);

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            0.f, wavelength_sample, adjusted_pos, aperture_sample);
        // Log(Debug, "Ray: %s", ray);
        sample(scene, sampler, film, pos, ray, block, aovs, active);
    }

protected:
    float m_max_time;
    float m_speed_of_sound;

    bool m_skip_direct;
    bool m_verbose;
};

MI_IMPLEMENT_CLASS_VARIANT(AcousticPathIntegrator, MonteCarloIntegrator)
MI_EXPORT_PLUGIN(AcousticPathIntegrator, "Acoustic Path Tracer integrator");
NAMESPACE_END(mitsuba)
