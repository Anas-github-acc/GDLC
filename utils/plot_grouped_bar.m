function plot_grouped_bar(groupNames, seriesA, seriesB, yLabelText, titleText, outPath, yLimits)
%PLOT_GROUPED_BAR  Two-series grouped bar chart, PowerPoint-friendly.
%
%   plot_grouped_bar(groupNames, paperValues, reproducedValues, ...
%                    yLabel, title, outputPngPath, yLimits)
%
%   Added by the Group 24 reproduction study.

if nargin < 7, yLimits = []; end

f = figure('Visible','off','Color','w','Position',[100 100 900 520]);
data = [seriesA(:), seriesB(:)];
h = bar(data, 'grouped');
set(h(1),'FaceColor',[0.20 0.40 0.70]);
set(h(2),'FaceColor',[0.90 0.55 0.15]);
grid on; box on;
set(gca,'XTickLabel',groupNames,'FontSize',12,'TickLabelInterpreter','none');
ylabel(yLabelText,'FontSize',13);
title(titleText,'FontSize',15,'FontWeight','bold');
legend({'Paper','Reproduced'},'Location','best','FontSize',12);
if ~isempty(yLimits)
    ylim(yLimits);
else
    ylim([0, max(data(:))*1.25 + eps]);
end

% Value labels above each bar
nGroups = size(data,1);
nSeries = size(data,2);
offset  = 0.40;   % MATLAB grouped bars for 2 series sit at g +- 0.2
for s = 1:nSeries
    for g = 1:nGroups
        xPos = g + (s - 1.5) * offset;
        text(xPos, data(g,s), sprintf('%.3f', data(g,s)), ...
             'HorizontalAlignment','center','VerticalAlignment','bottom', ...
             'FontSize',10,'FontWeight','bold');
    end
end

saveFigurePNG(f, outPath);
end
