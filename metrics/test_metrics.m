function test_metrics()
%TEST_METRICS  Sanity checks for adjusted_rand_index and clustering_purity.
%
%   Run:  addpath('metrics'); test_metrics
%
%   Prints PASS/FAIL for each case.  Added by the Group 24 reproduction study.

addpath(fullfile(fileparts(mfilename('fullpath'))));
tol = 1e-10;
nPass = 0; nFail = 0;

    function check(name, got, want)
        if isnan(want) && isnan(got)
            ok = true;
        else
            ok = abs(got - want) < 1e-6;
        end
        if ok
            fprintf('PASS  %-52s (got %.6f)\n', name, got);
            nPass = nPass + 1;
        else
            fprintf('FAIL  %-52s (got %.6f, expected %.6f)\n', name, got, want);
            nFail = nFail + 1;
        end
    end

% ---- ARI -------------------------------------------------------------- %
a = [1 1 2 2 3 3]';
check('ARI identical partitions', adjusted_rand_index(a,a), 1);

b = [7 7 9 9 4 4]';   % same partition, non-contiguous / permuted ids
check('ARI identical up to relabeling', adjusted_rand_index(a,b), 1);

c = [-5 -5 100 100 0 0]';   % arbitrary numeric (incl. negative) ids
check('ARI arbitrary numeric ids', adjusted_rand_index(a,c), 1);

d = ones(6,1);
check('ARI vs single-cluster partition', adjusted_rand_index(a,d), 0);

e = ones(6,1);
check('ARI both single-cluster (degenerate)', adjusted_rand_index(e,e), 1);

f = (1:6)';
check('ARI both singletons (degenerate)', adjusted_rand_index(f,f), 1);

% Known reference value: Hubert & Arabie worked example
t = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]';
p = [1 1 1 1 2 1 2 2 2 3 1 1 3 3 3]';
ariRef = referenceARI(t,p);
check('ARI matches brute-force pair counting', adjusted_rand_index(t,p), ariRef);

% ---- Purity ----------------------------------------------------------- %
check('Purity identical partitions', clustering_purity(a,a), 1);
check('Purity identical up to relabeling', clustering_purity(a,b), 1);
check('Purity single predicted cluster', clustering_purity(a,d), 2/6);

g = [1 1 1 1 2 2]';
h = [1 1 1 2 2 2]';
% cluster1 = {1,1,1} -> 3 ; cluster2 = {1,2,2} -> 2 ; total 5/6
check('Purity mixed clusters', clustering_purity(g,h), 5/6);

check('Purity all singletons', clustering_purity(a,f), 1);

fprintf('\n%d passed, %d failed.\n', nPass, nFail);
if nFail > 0
    warning('test_metrics:failures','%d metric test(s) failed.', nFail);
end
end

function ari = referenceARI(t,p)
% Brute-force ARI from raw pair counting, used only as a cross-check.
n = numel(t);
sameT = false(n); sameP = false(n);
for i = 1:n
    for j = 1:n
        sameT(i,j) = (t(i)==t(j));
        sameP(i,j) = (p(i)==p(j));
    end
end
mask = triu(true(n),1);
a = sum(sameT(mask) & sameP(mask));
b = sum(sameT(mask) & ~sameP(mask));
c = sum(~sameT(mask) & sameP(mask));
d = sum(~sameT(mask) & ~sameP(mask));
% Hubert-Arabie form
nc = a+b+c+d;
expected = (a+b)*(a+c)/nc;
ari = (a - expected) / (0.5*((a+b)+(a+c)) - expected);
end
