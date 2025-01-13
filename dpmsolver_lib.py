import torch
import torch.nn.functional as F
import math

################################################################################
# Minimal NoiseScheduleVP (discrete or linear)
################################################################################

class NoiseScheduleVP:
    def __init__(
        self,
        schedule='discrete',
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.,
        dtype=torch.float32,
    ):
        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule: {}".format(schedule))
        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        if self.schedule == 'discrete':
            return interpolate_fn(
                t.reshape((-1, 1)),
                self.t_array.to(t.device),
                self.log_alpha_array.to(t.device)
            ).reshape((-1))
        else:
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        else:
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1])
            )
            return t.reshape((-1,))

################################################################################
# Minimal model_wrapper (classifier-free, classifier, etc.)
################################################################################

def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    def get_model_input_time(t_continuous):
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return (x - expand_dims(alpha_t, x.dim()) * output) / expand_dims(sigma_t, x.dim())
        elif model_type == "v":
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return expand_dims(alpha_t, x.dim()) * output + expand_dims(sigma_t, x.dim()) * x
        else:
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output

    def cond_grad_fn(x, t_input):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        else:
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise_cond = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    return model_fn

################################################################################
# Minimal DPM_Solver (no dpmsolver++ or dynamic thresholding)
################################################################################

class DPM_Solver:
    def __init__(self, model_fn, noise_schedule):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule

    def noise_prediction_fn(self, x, t):
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        noise = self.noise_prediction_fn(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        return (x - sigma_t * noise) / alpha_t

    def model_fn(self, x, t):
        return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        else:
            t_order = 2
            t = torch.linspace(t_T**(1./t_order), t_0**(1./t_order), N+1).pow(t_order).to(device)
            return t

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3]*(K-2) + [2,1]
            elif steps % 3 == 1:
                orders = [3]*(K-1) + [1]
            else:
                orders = [3]*(K-1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2]*K
            else:
                K = steps // 2 + 1
                orders = [2]*(K-1) + [1]
        else:
            K = steps
            orders = [1]*steps
        if skip_type == 'logSNR':
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            arr = self.get_time_steps(skip_type, t_T, t_0, steps, device)
            idxs = torch.cumsum(torch.tensor([0]+orders), 0).to(device)
            timesteps_outer = arr[idxs]
        return timesteps_outer, orders

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = torch.expm1(h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x
            - (sigma_t * phi_1) * model_s
        )
        if return_intermediate:
            return x_t, {'model_s': model_s}
        return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("solver_type must be 'dpmsolver' or 'taylor'")
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1*h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s) * x
            - (sigma_s1 * phi_11) * model_s
        )
        model_s1 = self.model_fn(x_s1, s1)
        if solver_type == 'dpmsolver':
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
                - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
            )
        else:
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
                - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
            )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("solver_type must be 'dpmsolver' or 'taylor'")
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1*h
        lambda_s2 = lambda_s + r2*h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)
        phi_11 = torch.expm1(r1*h)
        phi_12 = torch.expm1(r2*h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2*h)/(r2*h) - 1.
        phi_2 = phi_1 / h - 1.
        phi_3 = phi_2 / h - 0.5
        if model_s is None:
            model_s = self.model_fn(x, s)
        if model_s1 is None:
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
        x_s2 = (
            torch.exp(log_alpha_s2 - log_alpha_s) * x
            - (sigma_s2 * phi_12) * model_s
            - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
        )
        model_s2 = self.model_fn(x_s2, s2)
        if solver_type == 'dpmsolver':
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
                - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
            )
        else:
            D1_0 = (1./r1)*(model_s1 - model_s)
            D1_1 = (1./r2)*(model_s2 - model_s)
            D1 = (r2*D1_0 - r1*D1_1)/(r2 - r1)
            D2 = 2.*(D1_1 - D1_0)/(r2 - r1)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("solver_type must be 'dpmsolver' or 'taylor'")
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1./r0)*(model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(h)
        if solver_type == 'dpmsolver':
            x_t = (
                torch.exp(log_alpha_t - log_alpha_prev_0) * x
                - (sigma_t * phi_1) * model_prev_0
                - 0.5 * (sigma_t * phi_1) * D1_0
            )
        else:
            x_t = (
                torch.exp(log_alpha_t - log_alpha_prev_0) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * (phi_1 / h - 1.)) * D1_0
            )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0/h, h_1/h
        D1_0 = (1./r0)*(model_prev_0 - model_prev_1)
        D1_1 = (1./r1)*(model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0/(r0+r1))*(D1_0 - D1_1)
        D2 = (1./(r0+r1))*(D1_0 - D1_1)
        phi_1 = torch.expm1(h)
        phi_2 = phi_1/h - 1.
        phi_3 = phi_2/h - 0.5
        x_t = (
            torch.exp(log_alpha_t - log_alpha_prev_0) * x
            - (sigma_t * phi_1) * model_prev_0
            - (sigma_t * phi_2) * D1
            - (sigma_t * phi_3) * D2
        )
        return x_t

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda xx, ss, tt: self.dpm_solver_first_update(xx, ss, tt, return_intermediate=True)
            higher_update = lambda xx, ss, tt, **kw: self.singlestep_dpm_solver_second_update(xx, ss, tt, r1=r1, solver_type=solver_type, **kw)
        else:
            r1, r2 = 1./3., 2./3.
            lower_update = lambda xx, ss, tt: self.singlestep_dpm_solver_second_update(xx, ss, tt, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda xx, ss, tt, **kw: self.singlestep_dpm_solver_third_update(xx, ss, tt, r1=r1, r2=r2, solver_type=solver_type, **kw)
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x)*atol, rtol*torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            def norm_fn(v):
                return torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta*h*torch.float_power(E, -1./order).float(), lambda_0 - lambda_s)
            nfe += order
        return x

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False):
        t_0 = 1./self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if return_intermediate:
                    intermediates.append(x)
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i+1]
                        model_prev_list[i] = model_prev_list[i+1]
                    t_prev_list[-1] = t
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                else:
                    K = steps // order
                    orders = [order]*K
                    timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device=device)
                for step, o in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step+1]
                    ts_inner = self.get_time_steps(skip_type, s.item(), t.item(), o, device=device)
                    lam_inner = self.noise_schedule.marginal_lambda(ts_inner)
                    h = lam_inner[-1] - lam_inner[0]
                    r1 = None if o<=1 else (lam_inner[1] - lam_inner[0]) / h
                    r2 = None if o<=2 else (lam_inner[2] - lam_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, o, solver_type=solver_type, r1=r1, r2=r2)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("method must be singlestep/multistep/singlestep_fixed/adaptive")
            if denoise_to_zero:
                t = torch.ones((1,)).to(device)*t_0
                x0 = self.data_prediction_fn(x, t)
                x = x0
                if return_intermediate:
                    intermediates.append(x)
        return (x, intermediates) if return_intermediate else x

def interpolate_fn(x, xp, yp):
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N,1,1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K-2, device=x.device), cand_start_idx)
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx+2, start_idx+1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K-2, device=x.device), cand_start_idx)
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2+1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x)*(end_y - start_y)/(end_x - start_x)
    return cand

def expand_dims(v, dims):
    return v[(...,) + (None,)*(dims - 1)]
