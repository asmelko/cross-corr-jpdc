if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU not available. Stopping.**\n']);
    return;
else
    dev = gpuDevice;
    fprintf(...
    'GPU detected (%s, %d multiprocessors, Compute Capability %s)',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
end

cpp_xcor = readmatrix('../outputs/out_cpp.csv', 'NumHeaderLines', 1);

in1 = readmatrix('../outputs/in1.csv', 'NumHeaderLines', 1);
in2 = readmatrix('../outputs/in2.csv', 'NumHeaderLines', 1);

g_in1 = gpuArray(in1);
g_in2 = gpuArray(in2);

g_xcor = xcorr2(g_in1, g_in2);

writematrix(gather(g_xcor), '../outputs/out_matlab.csv')

g_diff = g_xcor - gpuArray(cpp_xcor);

mean2(g_diff)
std2(g_diff)