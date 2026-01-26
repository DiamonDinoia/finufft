% test_pswf_beta.m
% For each (sigma,ns) in sigma_ns_beta0_beta.csv, evaluate error at the PSWF
% beta (first zero of prolate FT) and compare to a local-refined beta found
% by a coarse sweep + golden-section refinement (no reading of txt results).
%
% Usage: run('matlab/test/test_pswf_beta.m') in MATLAB (with FINUFFT MATLAB mex on path).

addpath(fileparts(mfilename('fullpath')))

% Params
prec = 'double';
myrand = @rand;
M = 1e3; Ntot = 300; ntr = 10; isign = +1;
o.showwarn = 0; warning('off','FINUFFT:epsTooSmall');
o.spread_kerformula = 8;   % PSWF tuned
dims = [true, false, false];

% Files
csvfile = fullfile(fileparts(mfilename('fullpath')),'..','..','sigma_ns_beta0_beta.csv');
out = fullfile(fileparts(mfilename('fullpath')),'results','pswf_beta_check_1D_double.txt');

T = readtable(csvfile);

% Open output
fout = fopen(out,'w');
fprintf(fout,'# sigma ns beta_pswf err_pswf beta_refined err_refined err_ratio_pswf\n');

for i = 1:height(T)
  sigma = T.sigma(i);
  ns = T.ns(i);
  beta_pswf = T.beta(i);

  % set options
  o.upsampfac = sigma;

  % find tol that yields desired kernel width ns
  [tol_w, wgot] = bisect_tol_for_w(ns, 10*eps(prec), 1.0, o);
  if isnan(tol_w) || wgot~=ns
    fprintf('Skipping sigma=%.3g ns=%d: cannot find tol->w\n',sigma,ns);
    continue
  end

  % evaluate PSWF beta error
  err_pswf = cached_eval_err(M,Ntot,ntr,isign,prec,tol_w,o,myrand,dims,beta_pswf);

  % compute refined beta via coarse sweep + golden refinement (local)
  beta_lo = max(0.5*beta_pswf, 0.1);
  beta_hi = min(1.5*beta_pswf, pi * (double(ns)) * (1.0 - 1.0/(2.0*sigma)) * 0.999);
  if beta_hi <= beta_lo
    beta_hi = max(beta_lo*1.1, beta_lo + 1e-3);
  end
  [beta_ref, err_ref] = beta_sweep_refine_simple(M,Ntot,ntr,isign,prec,tol_w,o,myrand,dims,beta_lo,beta_hi,beta_pswf);

  % ratio
  if err_ref==0 || isnan(err_ref)
    r = NaN;
  else
    r = err_pswf / err_ref;
  end

  fprintf(fout,'%.6g %d %.12g %.12g %.12g %.12g %.6g\n', sigma, ns, beta_pswf, err_pswf, beta_ref, err_ref, r);
  fprintf('sigma=%.3g ns=%d: err_pswf=%.3g, err_ref=%.3g, ratio=%.3g\n', sigma, ns, err_pswf, err_ref, r);
end

fclose(fout);
fprintf('Wrote results to %s\n',out);

% --------------------------
% local helper: coarse sweep + golden refinement (original simple approach)
function [beta_best, err_best] = beta_sweep_refine_simple(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta_lo,beta_hi,beta0)
  ngrid = 41;
  betas = linspace(beta_lo, beta_hi, ngrid);
  if beta0 > beta_lo && beta0 < beta_hi
    betas = unique([betas, beta0]);
  end
  errs = zeros(size(betas));
  for j = 1:numel(betas)
    errs(j) = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,betas(j));
  end
  [err_best, ibest] = min(errs);
  beta_best = betas(ibest);

  % golden-section refinement around the best neighbor bracket
  ilo = max(1, ibest-1);
  ihi = min(numel(betas), ibest+1);
  a = betas(ilo);
  b = betas(ihi);
  if b <= a
    return
  end
  phi = (1 + sqrt(5)) / 2;
  invphi = 1 / phi;
  invphi2 = 1 / (phi * phi);
  c = a + invphi2 * (b - a);
  d = a + invphi * (b - a);
  fc = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,c);
  fd = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,d);
  for it = 1:25
    if fc < fd
      b = d; d = c; fd = fc;
      c = a + invphi2 * (b - a);
      fc = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,c);
    else
      a = c; c = d; fc = fd;
      d = a + invphi * (b - a);
      fd = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,d);
    end
    if abs(b - a) < 1e-8 * (beta_hi - beta_lo)
      break
    end
  end
  [lerr, lidx] = min([fc, fd]);
  lvals = [c, d];
  if lerr < err_best
    beta_best = lvals(lidx);
    err_best = lerr;
  end
end

% --------------------------
% Helper functions (cached eval / eval_err / tol->w), compatible with betasweep
function e = cached_eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta)
  persistent cmap
  if isempty(cmap)
    cmap = containers.Map('KeyType','char','ValueType','double');
  end
  key = num2str([beta,tol,o.upsampfac,o.spread_kerformula],17);
  if isKey(cmap,key)
    e = cmap(key); return
  end
  e = eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta);
  cmap(key) = e;
end

function err = eval_err(M,Ntot,ntr,isign,prec,tol,o,myrand,dims,beta)
  oo = o; oo.spread_beta = beta;
  [nineerrs, ~] = erralltypedim(M,Ntot,ntr,isign,prec,tol,oo,myrand,dims);
  err = nineerrs(1,1);  % 1D type-1 error
end

function w = get_w_for_tol(tol,o)
  oo = o; oo.spreadinterponly = 1;
  du = finufft1d1(0.0,1.0,+1,tol,100,oo);
  w = sum(du~=0.0);
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
      tol_lo = tol_mid;
    else
      tol_hi = tol_mid;
    end
    tol = tol_mid; w = w_mid;
  end
end