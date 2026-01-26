function y = pswf(c, z)
% pswf(c,z) -- normalized prolate spheroidal window psi0^c(z)/psi0^c(0)
% vectorized in z
y = zeros(size(z));
for i = 1:numel(z)
  xi = z(i);
  if abs(xi) > 1
    y(i) = 0;
  else
    y(i) = prolate0_eval(c, xi) / prolate0_eval(c, 0);
  end
end
end

function val = prolate0_eval(c, x)
% cached workarray per c
persistent cache
key = sprintf('%.12g', c);
if isempty(cache), cache = containers.Map; end
if isKey(cache, key)
  w = cache(key);
else
  [ier, w] = prol0ini(c, 10000);
  if ier ~= 0
    error('prol0ini failed for c=%g (ier=%d)', c, ier);
  end
  cache(key) = w;
end
[val, ~] = prol0eva(x, w);
end

function [ier, w] = prol0ini(c, lenw)
% Build work array for prol0eva (port of Python/devel implementation).
% Minimal faithful port for inside [-1,1] evaluation.
w = zeros(1, lenw);
ier = 0;
thresh = 45.0;
iw = 11;
w(1) = iw + 0.1;
w(9) = thresh;

[ier, coeffs, nterms, rkhi] = prolps0i(c, lenw - iw);
if ier ~= 0
  return
end

w(iw:iw+numel(coeffs)-1) = coeffs;

if c >= thresh - 1e-10
  w(8) = c;
  w(5) = nterms + 0.1;
  return
end

ngauss = nterms * 2;
its = iw + (nterms + 2);
iwhts = its + (ngauss + 2);
ifs = iwhts + (ngauss + 2);

% check bounds
if ifs + ngauss + 2 - 1 > lenw
  ier = 1024; return
end

ts = legerts(ngauss);
[w(its:its+ngauss-1)] = ts;
% compute weights for nodes too
[~, whts] = legerts(ngauss);
w(iwhts:iwhts+ngauss-1) = whts;

% evaluate at nodes using Legendre expansion
pexp = w(iw:iw+nterms-1);
fs = zeros(1, ngauss);
for ii = 1:ngauss
  fs(ii) = legeexev(w(its+ii-1), pexp);
end
w(ifs:ifs+ngauss-1) = fs;

% eigenvalue rlam via prosinin at x0=0
f0 = legeexev(0.0, pexp);
[rlam, ~] = prosinin(c, w(its:its+ngauss-1), w(iwhts:iwhts+ngauss-1), fs, 0.0);
rlam = rlam / f0;

w(iw+3) = nterms + 0.1;
w(iw+4) = ngauss + 0.1;
w(iw+5) = rlam;
w(iw+6) = c;
end

function [ier, coeffs, nterms, rkhi] = prolps0i(c, lenw)
% wrapper port calling prolfun0
n = max(48, round(c*1.5)); % heuristic
[ier, coeffs, nterms, rkhi] = prolfun0(n, c, 1e-16);
end

function [ier, coeffs, nterms, rkhi] = prolfun0(n, c, epsv)
% Build Legendre expansion coefficients for psi0^c on [-1,1]
ier = 0;
delta = 1e-8;
ifsymm = 1;
ifodd = -1;

[as_, bs_, cs_] = prolmatr(n, c, 0.0, ifsymm, ifodd);

% QL on half
d = bs_;
e = as_;
ierr = prolql1(floor(n/2), d, e);
if ierr ~= 0
  ier = 2048; coeffs=[]; nterms=0; rkhi=0; return
end

rkhi = -d(floor(n/2));
rlam = rkhi + delta;

[as_, bs_, cs_] = prolmatr(n, c, rlam, ifsymm, ifodd);

% factorize on size n/2 using (bs, cs, as)
a = bs_(1:floor(n/2));
b = cs_(1:floor(n/2));
cc = as_(1:floor(n/2));
[u,v,wfac] = prolfact(a,b,cc);

% inverse iteration
xk = ones(floor(n/2),1);
numit = 4;
for kk=1:numit
  rhs = xk;
  prolsolv(u,v,wfac,rhs);
  dnorm = sqrt(sum(rhs.^2));
  rhs = rhs / dnorm;
  xk = rhs;
end

nterms = 0;
cs_out = zeros(floor(n/2),1);
for ii=1:floor(n/2)
  if abs(xk(ii))>epsv
    nterms = ii;
  end
  xk(ii) = xk(ii) * sqrt((ii-1)*2 + 0.5);
  cs_out(ii) = xk(ii);
end

% interleave even terms with zeros
coeffs = zeros(1, 2*(nterms+1));
for ii=1:(nterms+1)
  coeffs(2*ii-1) = cs_out(ii);
  coeffs(2*ii) = 0;
end
nterms = nterms * 2;
end

function [as_, bs_, cs_] = prolmatr(n, c, rlam, ifsymm, ifodd)
% build tri-diagonal representation
as_ = zeros(1,n);
bs_ = zeros(1,n);
cs_ = zeros(1,n);
for k0 = (ifodd>0)*1: (ifodd>0)*(2) + (ifodd<=0)*0 : (n+2)
  % loop handled below simpler: replicate Python prolcoef for k=0..n-1
end
% Use k from 0..n-1 in steps of 1 but adapt like Python did
kcount = 0;
if ifodd > 0
  klist = 1:2:(n*2);
else
  klist = 0:2:(n*2);
end
for idx = 1:length(klist)
  k0 = klist(idx);
  k = idx;
  [uk,vk,wk,alpha,beta,gamma] = prolcoef(rlam, k0, c);
  as_(k) = alpha;
  bs_(k) = beta;
  cs_(k) = gamma;
  if ifsymm ~= 0
    if ifodd > 0
      if k0 > 1
        as_(k) = as_(k) / sqrt(k0 - 2 + 0.5) * sqrt(k0 + 0.5);
      end
      cs_(k) = cs_(k) * sqrt(k0 + 0.5) / sqrt(k0 + 0.5 + 2);
    else
      if k0 ~= 0
        as_(k) = as_(k) / sqrt(k0 - 2 + 0.5) * sqrt(k0 + 0.5);
      end
      cs_(k) = cs_(k) * sqrt(k0 + 0.5) / sqrt(k0 + 0.5 + 2);
    end
  end
end
end

function [uk,vk,wk,alpha,beta,gamma] = prolcoef(rlam, k, c)
d = k*(k-1) / ((2*k+1)*(2*k-1));
uk = d;
d1 = (k+1)^2 / (2*k+3);
d2 = k^2 / (2*k-1);
vk = (d1 + d2) / (2*k+1);
wk = ( (k+1)*(k+2) ) / ((2*k+1)*(2*k+3));
alpha = -c^2 * uk;
beta  = rlam - k*(k+1) - c^2 * vk;
gamma = -c^2 * wk;
end

function ierr = prolql1(n, d, e)
% in-place QL for symmetric tridiagonal
ierr = 0;
if n<=1, return; end
% shift e down (MATLAB indexing)
for i=1:n-1, e(i) = e(i+1); end
e(n)=0;
for l=1:n
  j=0;
  while true
    m = [];
    for mm = l:n-1
      tst1 = abs(d(mm)) + abs(d(mm+1));
      tst2 = tst1 + abs(e(mm));
      if tst2 == tst1
        m = mm; break
      end
    end
    if isempty(m), m = n; end
    if m == l, break; end
    if j == 30, ierr = l; return; end
    j = j + 1;
    g = (d(l+1)-d(l)) / (2*e(l));
    r = sqrt(g*g + 1);
    g = d(m) - d(l) + e(l) / (g + sign(r)*g);
    s = 1; c = 1; p = 0;
    for i = m-1:-1:l
      f = s*e(i);
      b = c*e(i);
      r = sqrt(f*f + g*g);
      e(i+1) = r;
      if r == 0
        d(i+1) = d(i+1) - p;
        e(m) = 0; break
      end
      s = f/r; c = g/r;
      g = d(i+1) - p;
      r = (d(i) - g)*s + 2*c*b;
      p = s*r;
      d(i+1) = g + p;
      g = c*r - b;
    end
    if r == 0, break; end
    d(l) = d(l) - p;
    e(l) = g;
    e(m) = 0;
  end
  if l ~= 1
    for i = l:-1:2
      if d(i) >= d(i-1), break; end
      tmp = d(i); d(i)=d(i-1); d(i-1)=tmp;
    end
  end
end
end

function [u,v,w] = prolfact(a,b,c)
n = numel(a);
aa = a;
u = zeros(n,1); v = zeros(n,1); w = zeros(n,1);
for i=1:n-1
  d = c(i+1)/aa(i);
  aa(i+1) = aa(i+1) - b(i)*d;
  u(i) = d;
end
for i=n:-1:2
  d = b(i-1)/aa(i);
  v(i) = d;
end
for i=1:n
  w(i) = 1/aa(i);
end
end

function prolsolv(u,v,wfac,rhs)
n = numel(rhs);
for i=1:n-1
  rhs(i+1) = rhs(i+1) - u(i)*rhs(i);
end
for i=n:-1:2
  rhs(i-1) = rhs(i-1) - rhs(i)*v(i);
end
rhs(:) = rhs(:) .* wfac(:);
end

function [rint, derrint] = prosinin(c, ts, whts, fs, x)
rint = 0; derrint = 0;
for i=1:numel(ts)
  diff = x - ts(i);
  if diff == 0, diff = 1e-300; end
  sin_term = sin(c*diff); cos_term = cos(c*diff);
  rint = rint + whts(i)*fs(i)*sin_term/diff;
  derrint = derrint + whts(i)*fs(i)/(diff*diff) * (c*diff*cos_term - sin_term);
end
end

function val = legeexev(x, pexp)
n = numel(pexp)-1;
if n < 0, val = 0; return; end
if n == 0, val = pexp(1); return; end
pjm2 = 1; pjm1 = x;
val = pexp(1)*pjm2 + pexp(2)*pjm1;
for j=2:n
  pj = ((2*j-1)*x*pjm1 - (j-1)*pjm2)/j;
  val = val + pexp(j+1)*pj;
  pjm2 = pjm1; pjm1 = pj;
end
end

function [val, der] = legeFDER(x, pexp)
n = numel(pexp)-1;
if n<0, val=0; der=0; return; end
if n==0, val=pexp(1); der=0; return; end
if n==1, val=pexp(1)+pexp(2)*x; der=pexp(2); return; end
pjm2=1; pjm1=x;
derjm2=0; derjm1=1;
val = pexp(1)*pjm2 + pexp(2)*pjm1;
der = pexp(2);
for j=2:n
  pj = ((2*j-1)*x*pjm1 - (j-1)*pjm2)/j;
  val = val + pexp(j+1)*pj;
  derj = (2*j-1)*(pjm1 + x*derjm1) - (j-1)*derjm2;
  derj = derj / j;
  der = der + pexp(j+1)*derj;
  pjm2 = pjm1; pjm1 = pj;
  derjm2 = derjm1; derjm1 = derj;
end
end

function ts = legerts(n)
% Return Gauss-Legendre roots on [-1,1] (approx via built-in roots of Legendre)
% Use built-in: roots of Legendre can be approximated via eigenvalues of Jacobi matrix.
if n <= 0, ts = []; return; end
beta = 0.5./sqrt(1 - (2*(1:n-1)).^(-2));
J = diag(zeros(n,1)) + diag(beta,1) + diag(beta,-1);
[V, D] = eig(J);
ts = diag(D);
ts = sort(ts)';
end