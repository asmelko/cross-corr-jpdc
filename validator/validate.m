if ~(parallel.gpu.GPUDevice.isAvailable)
    fprintf(['\n\t**GPU not available. Stopping.**\n']);
    return;
else
    dev = gpuDevice;
    fprintf(...
    'GPU detected (%s, %d multiprocessors, Compute Capability %s)',...
    dev.Name, dev.MultiprocessorCount, dev.ComputeCapability);
end

cpp_xcor = readmatrix('cpp_xcor.csv');

in1 = readmatrix('in1.csv');
in2 = readmatrix('in2.csv');

g_in1 = gpuArray(in1);
g_in2 = gpuArray(in2);

g_xcor = xcorr2(g_in1, g_in2);

g_diff = g_xcor - gpuArray(cpp_xcor);

mean2(g_diff)
std2(g_diff)