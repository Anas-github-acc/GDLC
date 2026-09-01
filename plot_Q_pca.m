%PLOT_Q_PCA  Optional PCA visualisation of the learned GDLC representation.
%
%   Group 24 reproduction study.  Requires run_reproduction to have been run
%   first (it saves results/BASEHOCK_repro_extra.mat containing extra.Q).
%
%   Ground-truth labels are used ONLY to colour the scatter plot; they play
%   no part in learning Q.
%
%   PCA is computed here from the SVD of the centred data, so no Statistics
%   Toolbox is required.
%
%   Usage:   clear; clc; close all; plot_Q_pca

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(fullfile(thisDir,'utils'));

matFile = fullfile(thisDir,'results','BASEHOCK_repro_extra.mat');
if ~exist(matFile,'file')
    error('plot_Q_pca:missingInput', ...
          'Run run_reproduction first -- %s not found.', matFile);
end
L = load(matFile);
Q   = L.BASEHOCK_extra.Q;
gnd = double(L.BASEHOCK_gnd(:));

extFiguresDir = fullfile(thisDir,'figures','extensions');
if ~exist(extFiguresDir,'dir'), mkdir(extFiguresDir); end

% ---- PCA via SVD of the centred matrix --------------------------------- %
Qc = bsxfun(@minus, Q, mean(Q,1));
[U,Sg,~] = svd(Qc, 'econ');
scores = U * Sg;
sv = diag(Sg);
varExp = sv.^2 / sum(sv.^2) * 100;

nComp = size(scores,2);
pc1 = scores(:,1);
if nComp >= 2
    pc2 = scores(:,2);
else
    pc2 = zeros(size(pc1));   % C = 1 -> plot against a constant axis
end

classes = unique(gnd);
colors  = lines(numel(classes));

f = figure('Visible','off','Color','w','Position',[100 100 860 620]);
hold on;
for c = 1:numel(classes)
    idx = (gnd == classes(c));
    scatter(pc1(idx), pc2(idx), 18, colors(c,:), 'filled', ...
            'MarkerFaceAlpha', 0.6);
end
grid on; box on;
if nComp >= 2
    xlabel(sprintf('PC1 (%.1f%% var)', varExp(1)),'FontSize',13);
    ylabel(sprintf('PC2 (%.1f%% var)', varExp(2)),'FontSize',13);
else
    xlabel(sprintf('PC1 (%.1f%% var)', varExp(1)),'FontSize',13);
    ylabel('(only one component available)','FontSize',13);
end
title('BASEHOCK -- PCA of learned GDLC representation Q', ...
      'FontSize',15,'FontWeight','bold');
legend(arrayfun(@(c) sprintf('True class %d', c), classes, ...
       'UniformOutput', false), 'Location','best','FontSize',12);
set(gca,'FontSize',12);
saveFigurePNG(f, fullfile(extFiguresDir,'BASEHOCK_Q_PCA.png'));

fprintf('PCA visualisation written.\n');
