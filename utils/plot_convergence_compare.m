function plot_convergence_compare(res, fieldName, yLabelText, titleText, outPath)
%PLOT_CONVERGENCE_COMPARE  Overlay a per-round metric for several variants.
%
%   res must be a struct array with fields .name and the metric field named
%   by fieldName (a 1 x nRounds vector).
%
%   Added by the Group 24 reproduction study.

markers = {'-o','-s','-^','-d','-v','-p'};
f = figure('Visible','off','Color','w','Position',[100 100 880 530]);
hold on;
for e = 1:numel(res)
    y = res(e).(fieldName);
    plot(1:numel(y), y, markers{mod(e-1,numel(markers))+1}, ...
         'LineWidth',2,'MarkerSize',7);
end
grid on; box on;
xlabel('GDLC Round','FontSize',13);
ylabel(yLabelText,'FontSize',13);
title(titleText,'FontSize',15,'FontWeight','bold','Interpreter','none');
legend({res.name},'Location','southeast','FontSize',12,'Interpreter','none');
set(gca,'FontSize',12,'XTick',1:numel(res(1).(fieldName)));
ylim([0 1.05]);
saveFigurePNG(f, outPath);
end
