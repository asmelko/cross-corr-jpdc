if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU not available. Stopping.**\n']);
    return;
else
    dev = gpuDevice;
    fprintf(...
    'GPU detected (%s, %d multiprocessors, Compute Capability %s)\n',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
end

fprintf(...
    'Computing cross-correlation of "%s" and "%s", writing results to "%s"\n',...
    in1_path, in2_path, out_path);

% in1_path and in2_path set externally
in1 = readmatrix(in1_path, 'NumHeaderLines', 1);
in2 = readmatrix(in2_path, 'NumHeaderLines', 1);

g_in1 = gpuArray(in1);
g_in2 = gpuArray(in2);

g_xcor = xcorr2(g_in1, g_in2);

host_result = gather(g_xcor);
% out_path set externally
writematrix(size(host_result),out_path, "Delimiter", ",", "FileType", "text")
writematrix(host_result, out_path, "Delimiter", ",", "FileType", "text", "WriteMode", "append")

disp('Done')
