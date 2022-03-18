% Fourier domain Huffman coding with cost per bits
% Name: Zhenhuan Sun
% Student ID: 20735233

%%
clear
clc
close all

% Read image that need to be compressed
x = imread("lena_std.tiff");
x = rgb2gray(x);
[m, n] = size(x);
x_binary = de2bi(x);
xe_len = numel(x_binary);

% DCT compression
x_double = im2double(x);
T = dctmtx(8);
dct = @(block_struct) T * block_struct.data * T';
B = blockproc(x_double,[8 8],dct);
mask = [1   1   1   1   0   0   0   0
        1   1   1   0   0   0   0   0
        1   1   0   0   0   0   0   0
        1   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0];
B2 = blockproc(B,[8 8],@(block_struct) mask .* block_struct.data);
invdct = @(block_struct) T' * block_struct.data * T;
I2 = blockproc(B2,[8 8],invdct);
I2_binary = de2bi(uint8(I2));
I2_binary_length = numel(I2_binary);
% figure(1)
% subplot(1, 2, 1); imshow(x);
% subplot(1, 2, 2); imshow(uint8(I2));

% Standard Huffman Coding to original image
h_std = hist(x(:), 0:255);
p_std = h_std / numel(x);
dict_std = huffmandict(0:255, p_std);
code_std = huffmanenco(x(:), dict_std);
binary_code_std = de2bi(code_std);
encoded_length_std = numel(binary_code_std);

% separate image to magnitude and phase components
x_dft = fftshift(fft2(x));
% magnitude
x_dft_mag = abs(x_dft);
x_dft_mag_ln = log(x_dft_mag+1);
x_dft_mag_norm = log(x_dft_mag+1)/max(x_dft_mag_ln(:));
% phase
x_dft_phase = angle(x_dft);

% Apply square low-pass filter to both magnitude and phase
x_dft_mag_lp = zeros(m, n);
x_dft_phase_lp = zeros(m, n);
[C, R] = meshgrid(1:m, 1:n);
scale = 8;
lp = abs(R - m/2) <= m/scale & abs(C - n/2) <= n/scale;
x_dft_mag_lp(lp) = x_dft_mag_norm(lp);
x_dft_phase_lp(lp) = x_dft_phase(lp);

% Crop lowpassed part
r1 = m;
r = 2 * m/scale;
c1 = n;
c = 2 * n/scale;
x_dft_mag_lp_cropped_temp = x_dft_mag_lp(round((r1-r)/2)+1:round((r1+r)/2), round((c1-c)/2)+1:round((c1+c)/2));
x_dft_mag_lp_cropped = uint8(x_dft_mag_lp_cropped_temp * 255); 
x_dft_phase_lp_cropped_temp = x_dft_phase_lp(round((r1-r)/2)+1:round((r1+r)/2), round((c1-c)/2)+1:round((c1+c)/2));
min_phase = min(x_dft_phase_lp_cropped_temp(:));
temp_matrix = x_dft_phase_lp_cropped_temp + abs(min_phase);
max_temp_matrix = max(temp_matrix(:));
temp_matrix_norm = temp_matrix / max_temp_matrix;
x_dft_phase_lp_cropped = uint8(temp_matrix_norm * 255);

% Huffman encoding with cost per bits
%h = hist(x_dft_mag_lp_cropped_temp(:), unique(x_dft_phase_lp_cropped_temp(:)));
h_mag = hist(x_dft_mag_lp_cropped(:), 0:255);
p_mag = h_mag / numel(x_dft_mag_lp_cropped);
i = 1:256;
mean = sum(i .* p_mag);
sigma = sqrt(sum(i.^2 .* p_mag) - mean^2);
% mean = 256;
% sigma = 10;
costs_mag = 95/sqrt(2*pi)*sigma * exp(-1/2 * ((i-mean)/sigma).^2);
temp_mag = p_mag .* costs_mag;
q_mag = temp_mag ./ sum(temp_mag(:));
figure(1)
hist(x_dft_mag_lp_cropped(:), 0:255);
hold on
plot(i, costs_mag);
hold off
h_phase = hist(x_dft_phase_lp_cropped(:), 0:255);
p_phase = h_phase / numel(x_dft_phase_lp_cropped);
costs_phase = 80 * 256 / 256 * ones(1, 256); % normal distribution
temp_phase = p_phase .* costs_phase;
q_phase = temp_phase ./ sum(temp_phase(:));
figure(2)
hist(x_dft_phase_lp_cropped(:), 0:255);
hold on
plot(i, costs_phase);
hold off
dict_q_mag = huffmandict(0:255, q_mag);
dict_q_phase = huffmandict(0:255, q_phase);
% dict_p_mag = huffmandict(0:255, p_mag);
% dict_p_phase = huffmandict(0:255, p_phase);
code_mag = huffmanenco(x_dft_mag_lp_cropped(:), dict_q_mag);
code_phase = huffmanenco(x_dft_phase_lp_cropped(:), dict_q_phase);
% code_mag = huffmanenco(x_dft_mag_lp_cropped(:), dict_p_mag);
% code_phase = huffmanenco(x_dft_phase_lp_cropped(:), dict_p_mag);
binary_code_mag = de2bi(code_mag);
encoded_length_mag = numel(binary_code_mag);
binary_code_phase = de2bi(code_phase);
encoded_length_phase = numel(binary_code_phase);

% Huffman decoding with cost per bits
x_dft_mag_lp_recovered = reshape(huffmandeco(code_mag, dict_q_mag), [r, c]);
x_dft_phase_lp_recovered = reshape(huffmandeco(code_phase, dict_q_phase), [r, c]);
% x_dft_mag_lp_recovered = reshape(huffmandeco(code_mag, dict_p_mag), [r, c]);
% x_dft_phase_lp_recovered = reshape(huffmandeco(code_phase, dict_p_mag), [r, c]);
%isequal(x_dft_mag_lp_recovered, x_dft_mag_lp_cropped)
%isequal(x_dft_phase_lp_recovered, x_dft_phase_lp_cropped)
x_dft_mag_lp_recovered_reversed = double(x_dft_mag_lp_recovered) / 255;
x_dft_phase_lp_recovered_reversed = double(x_dft_phase_lp_recovered) / 255 * max_temp_matrix - abs(min_phase);
x_dft_mag_lp_recovered_reversed_resized = zeros(m, n, 'double'); 
x_dft_phase_lp_recovered_reversed_resized = zeros(m, n, 'double');
for i = 1:m
    for j = 1:n
        if abs(i-m/2)<m/scale && abs(j-n/2)<n/scale
            x_dft_mag_lp_recovered_reversed_resized(i, j) = x_dft_mag_lp_recovered_reversed(i-192, j-192);
            x_dft_phase_lp_recovered_reversed_resized(i, j) = x_dft_phase_lp_recovered_reversed(i-192, j-192);
        end
    end
end
x_dft_mag_lp_recovered_reversed_resized = exp(x_dft_mag_lp_recovered_reversed_resized * max(x_dft_mag_ln(:))) - 1;
x_reconstructed_mine = real(ifft2(ifftshift(x_dft_mag_lp_recovered_reversed_resized .* exp(1i * x_dft_phase_lp_recovered_reversed_resized))));
x_reconstructed = real(ifft2(ifftshift(x_dft_mag_lp .* exp(1i * x_dft_phase_lp))));

figure(3)
subplot(2, 2, 1); imshow(x_dft_phase_lp_cropped_temp, []); title('original');
subplot(2, 2, 2); imshow(x_dft_phase_lp_recovered_reversed, []); title('recovered');
subplot(2, 2, 3); imshow(x_reconstructed, []); title('recovered (real)');
subplot(2, 2, 4); imshow(x_reconstructed_mine, []); title('recovered (mine)');