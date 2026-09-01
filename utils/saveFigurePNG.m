function saveFigurePNG(figHandle, outPath)
%SAVEFIGUREPNG  Save a figure to PNG at presentation resolution and close it.
%
%   Uses exportgraphics when available (R2020a+), otherwise falls back to
%   print -dpng.  Added by the Group 24 reproduction study.

try
    if exist('exportgraphics','file') == 2 || exist('exportgraphics','builtin') == 5
        exportgraphics(figHandle, outPath, 'Resolution', 200);
    else
        set(figHandle,'PaperPositionMode','auto','InvertHardcopy','off');
        print(figHandle, outPath, '-dpng', '-r200');
    end
catch err
    warning('saveFigurePNG:failed','Could not save %s (%s)', outPath, err.message);
end
close(figHandle);
fprintf('  wrote %s\n', outPath);
end
