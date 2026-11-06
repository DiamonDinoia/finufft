% Tests M=0 (empty NU point sets) return zeros, as the C core does (issue 937).
% Covers the simple 1d/2d/3d interfaces, ntrans inference from c, and the plan.

ms = 6; mt = 4; mu = 3;
x = []; c = [];

% type 1 with c=[] gives zeros of mode size...
f = finufft1d1(x, c, 1, 1e-12, ms);
assert(isequal(size(f), [ms 1]) && ~any(f(:)));
f = finufft2d1(x, [], c, 1, 1e-12, ms, mt);
assert(isequal(size(f), [ms mt]) && ~any(f(:)));
f = finufft3d1(x, [], [], c, 1, 1e-12, ms, mt, mu);
assert(isequal(size(f), [ms mt mu]) && ~any(f(:)));

% ntrans=2 inferred from an empty c of size 0-by-2...
f = finufft1d1(x, zeros(0,2), 1, 1e-12, ms);
assert(isequal(size(f), [ms 2]) && ~any(f(:)));
f = finufft2d1(x, [], zeros(0,2), 1, 1e-12, ms, mt);
assert(isequal(size(f), [ms mt 2]) && ~any(f(:)));
f = finufft3d1(x, [], [], zeros(0,2), 1, 1e-12, ms, mt, mu);
assert(isequal(size(f), [ms mt mu 2]) && ~any(f(:)));

% type 2 gives empty output at empty targets...
f = randn(ms,1) + 1i*randn(ms,1);
cc = finufft1d2(x, 1, 1e-12, f);
assert(isempty(cc));

% type 3 with nonempty targets gives zeros at the targets...
s = 2*pi*rand(5,1);
ff = finufft1d3(x, c, 1, 1e-12, s);
assert(isequal(size(ff), [5 1]) && ~any(ff(:)));
ff = finufft3d3(x, [], [], c, 1, 1e-12, s, s, s);
assert(isequal(size(ff), [5 1]) && ~any(ff(:)));
ff = finufft1d3(x, zeros(0,2), 1, 1e-12, s);
assert(isequal(size(ff), [5 2]) && ~any(ff(:)));

% plan interface with empty points...
p = finufft_plan(1, ms, 1, 1, 1e-12);
p.setpts(x);
f = p.execute(c);
assert(isequal(size(f), [ms 1]) && ~any(f(:)));
delete(p);
p = finufft_plan(3, 1, 1, 1, 1e-12);
p.setpts(x, [], [], s, [], []);
ff = p.execute(c);
assert(isequal(size(ff), [5 1]) && ~any(ff(:)));
delete(p);

% empty x still selects the precision from class(x)...
fs = finufft1d1(single([]), single([]), 1, 1e-6, ms);
assert(isa(fs, 'single') && isequal(size(fs), [ms 1]) && ~any(fs(:)));

disp('empty_nupts_issue: all tests passed');
