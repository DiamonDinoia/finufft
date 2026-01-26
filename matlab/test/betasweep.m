% Sweep beta for PSWF tuned kernel (kf=8) in 1D, using default sizes.
% Uses tol bisection to hit target ns (kernel width) via spreadinterponly.
% Barnett-style test harness; outputs text results for offline fitting.

addpath(fileparts(mfilename('fullpath')))
clear

prec = 'double';
myrand = @rand;
text_only = true;

M = 1e3;             % # NU pts
dim = 1; Ntot = 300; % 1D defaults from wsweepkerrcomp
ntr = 10;            % #transforms
isign = +1;
sigmas = 1.25:0.05:2.0;
ns_list = 2:16;

o.upsampfac = sigmas(1);
o.showwarn = 0; warning('off','FINUFFT:epsTooSmall');
o.spread_kerformula = 3;   % PSWF tuned
dims = false(1, 3); dims(dim) = true;

if text_only
  outdir = 'results';
  if ~exist(outdir,'dir'), mkdir(outdir); end
  outfile = sprintf('%s/betasweep_text_%dD_%s.txt',outdir,dim,prec);
  fid = fopen(outfile,'w');
  fprintf(fid,'# betasweep text output: dim=%d prec=%s M=%d Ntot=%d ntr=%d kf=%d\n',...
          dim,prec,M,Ntot,ntr,o.spread_kerformula);
  fprintf(fid,'# sigma ns tol beta0 beta_best err_best err_ratio\n');
end

for s = 1:numel(sigmas)
  sigma = sigmas(s);
  o.upsampfac = sigma;
  for ns = ns_list
    tol_lo = 10 * eps(prec);  % small tol -> large w
    tol_hi = 1.0;             % large tol -> small w
    [tol_w, w_got] = bisect_tol_for_w(ns, tol_lo, tol_hi, o);
    if isnan(tol_w) || w_got ~= ns
      continue
    end

    % baseline beta from polynomial heuristic (prolate-derived starting point)
    t = (double(ns)) * (1.0 - 1.0 / (2.0 * sigma));
    beta0 = ((-0.00149087 * t + 0.0218459) * t + 3.06269) * t - 0.0365245;
    beta_cut = pi * t * 0.999;     % keep just below cutoff
    beta_lo = 0.5 * beta0;
    beta_hi = min(1.5 * beta0, beta_cut);
    if beta_hi <= beta_lo
      beta_hi = beta_cut;
      beta_lo = 0.5 * beta_hi;
    end

    % refine beta with robust multi-start local search
    [beta_best, err_best] = beta_sweep_refine(M,Ntot,ntr,isign,prec,tol_w,o,myrand,dims,beta_lo,beta_hi,beta0);
    err0 = cached_eval_err(M,Ntot,ntr,isign,prec,tol_w,o,myrand,dims,beta0);
    if err0 <= err_best
      beta_best = beta0;
      err_best = err0;
    end
    if text_only
      fprintf(fid,'%.6g %d %.6g %.6g %.6g %.6g %.6g\n',...
              sigma, ns, tol_w, beta0, beta_best, err_best, err_best/err0);
    end
  end
end

if text_only
  fclose(fid);
  fprintf('Wrote text results to %s\n',outfile);
end

function [beta_best, err_best] = beta_sweep_refine(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta_lo,beta_hi,beta0)
% Robust multi-start local refinement to avoid local minima.
% - coarse grid sampling to find promising seeds
% - run local fminbnd refinements (if available) from each seed, else golden fallback
% - use cached_eval_err to avoid duplicate expensive evaluations

ngrid = 21;
betas = linspace(beta_lo, beta_hi, ngrid);
% Ensure baseline beta evaluated
if beta0 > beta_lo && beta0 < beta_hi
  betas = unique([betas, beta0]);
end

errs = zeros(size(betas));
for i = 1:numel(betas)
  errs(i) = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,betas(i));
end

% pick up to k best grid points as multi-start seeds
k = min(3, numel(betas));
[~, order] = sort(errs);
seeds = betas(order(1:k));

best_err = inf;
best_beta = seeds(1);

range_beta = beta_hi - beta_lo;
local_step = max((betas(2)-betas(1))*3, 1e-8*range_beta);

for sidx = 1:numel(seeds)
  seed = seeds(sidx);
  a = max(beta_lo, seed - local_step);
  b = min(beta_hi, seed + local_step);
  if b <= a
    continue
  end

  f = @(x) cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,x);

  % initial local best at seed
  local_best_beta = seed;
  local_best_err = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,seed);

  % try Brent (fminbnd) for local refinement if available
  if exist('fminbnd','file')
    try
      opts = optimset('TolX',1e-6,'MaxFunEvals',200);
      [bopt, ~] = fminbnd(f, a, b, opts);
      bopt_err = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,bopt);
      if bopt_err < local_best_err
        local_best_beta = bopt;
        local_best_err = bopt_err;
      end
    catch
      % fall back to golden below
    end
  end

  % golden-section fallback for robustness
  phi = (1 + sqrt(5)) / 2;
  invphi = 1 / phi;
  invphi2 = 1 / (phi * phi);
  aa = a; bb = b;
  c = aa + invphi2 * (bb - aa);
  d = aa + invphi * (bb - aa);
  fc = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,c);
  fd = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,d);
  for it = 1:12
    if fc < fd
      bb = d; d = c; fd = fc;
      c = aa + invphi2 * (bb - aa);
      fc = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,c);
    else
      aa = c; c = d; fc = fd;
      d = aa + invphi * (bb - aa);
      fd = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,d);
    end
    if abs(bb - aa) < 1e-8 * range_beta
      break
    end
  end
  [lerr, lidx] = min([fc, fd]);
  lvals = [c, d];
  if lerr < local_best_err
    local_best_beta = lvals(lidx);
    local_best_err = lerr;
  end


  if local_best_err < best_err
    best_err = local_best_err;
    best_beta = local_best_beta;
  end
end

% fallback to best grid point if nothing improved
[grid_best_err, ig] = min(errs);
grid_best_beta = betas(ig);
if grid_best_err < best_err
  best_err = grid_best_err;
  best_beta = grid_best_beta;
end

beta_best = best_beta;
err_best = best_err;
end

function err = eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta)
oo = o;
oo.spread_beta = beta;
[nineerrs, ~] = erralltypedim(M,Ntot,ntr,isign,prec,tol,oo,myrand,dims);
err = nineerrs(1,1);  % 1D type-1 error
end

function e = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta)
% Persistent cache for eval_err to avoid duplicate expensive calls.
persistent cmap
if isempty(cmap)
  cmap = containers.Map('KeyType','char','ValueType','double');
end
key = num2str([beta,tol,o.upsampfac,o.spread_kerformula],17);
if isKey(cmap, key)
  e = cmap(key);
  return
end
e = eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta);
cmap(key) = e;
end

function w = get_w_for_tol(tol,o)
  persistent wcache
  if isempty(wcache)
    wcache = containers.Map('KeyType','char','ValueType','double');
  end
  key = num2str([tol,o.upsampfac,o.spread_kerformula],17);
  if isKey(wcache,key)
    w = wcache(key); return
  end
  oo = o; oo.spreadinterponly = 1;
  du = finufft1d1(0.0,1.0,+1,tol,100,oo);
  w = sum(du~=0.0);
  wcache(key) = w;
end

function [tol, w] = bisect_tol_for_w(wtarget,tol_lo,tol_hi,o)
max_iter = 40;
w_lo = get_w_for_tol(tol_lo,o);
w_hi = get_w_for_tol(tol_hi,o);
if wtarget < w_hi || wtarget > w_lo
  tol = nan; w = nan; return
end
tol = nan; w = nan;
for it = 1:max_iter
  tol_mid = sqrt(tol_lo * tol_hi);
  w_mid = get_w_for_tol(tol_mid,o);
  if w_mid == wtarget
    tol = tol_mid; w = w_mid; return
  elseif w_mid > wtarget
    tol_lo = tol_mid;   % tol too small -> w too big
  else
    tol_hi = tol_mid;   % tol too big -> w too small
  end
  tol = tol_mid; w = w_mid;
end
end