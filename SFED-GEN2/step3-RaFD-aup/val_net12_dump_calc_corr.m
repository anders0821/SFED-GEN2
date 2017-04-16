function [] = val_net12_dump_calc_corr()
clc;
clear;
close all;

% 加载数据
load 100-NET12-DUMP/dump.mat

% 预处理数据
N = size(VAL_X, 4);
VAL_X = single(VAL_X);
VAL_X = reshape(VAL_X, [64*64 N]);
VAL_X_HAT = reshape(VAL_X_HAT, [64*64 N]);
VAL_Z_HAT = reshape(VAL_Z_HAT, [64*64 N]);
VAL_X = VAL_X';
VAL_Z_HAT = VAL_Z_HAT';
VAL_Y_HAT = VAL_Y_HAT';
VAL_H2_HAT = VAL_H2_HAT';
VAL_X_HAT = VAL_X_HAT';

% 可视化
% VAL_Y_HAT_2 = reshape(VAL_Y_HAT, [N 1 7]);
% VAL_Y_HAT_2 = repmat(VAL_Y_HAT_2, [1 73 1]);
% VAL_Y_HAT_2 = reshape(VAL_Y_HAT_2, [N 73*7]);
% C = corr([VAL_Y_HAT_2 VAL_H2_HAT]);
% figure;
% imshow(abs(C), [0 1]);

% 可视化
% VAL_Y_HAT_2 = reshape(VAL_Y_HAT, [N 1 7]);
% VAL_Y_HAT_2 = repmat(VAL_Y_HAT_2, [1 585 1]);
% VAL_Y_HAT_2 = reshape(VAL_Y_HAT_2, [N 585*7]);
% C = corr([VAL_Y_HAT_2 VAL_Z_HAT]);
% figure;
% imshow(abs(C), [0 1]);

% 可视化
VAL_Y_HAT_2 = reshape(VAL_Y_HAT, [N 1 7]);
VAL_Y_HAT_2 = repmat(VAL_Y_HAT_2, [1 585 1]);
VAL_Y_HAT_2 = reshape(VAL_Y_HAT_2, [N 585*7]);
VAL_H2_HAT_2 = reshape(VAL_H2_HAT, [N 1 512]);
VAL_H2_HAT_2 = repmat(VAL_H2_HAT_2, [1 8 1]);
VAL_H2_HAT_2 = reshape(VAL_H2_HAT_2, [N 8*512]);
C = corr([VAL_Y_HAT_2 VAL_H2_HAT_2 VAL_Z_HAT]);

L1 = size(VAL_Y_HAT_2, 2);
L2 = size(VAL_H2_HAT_2, 2);
MARGIN = 100;
C = insertcols(C, L1+L2+1, MARGIN);
C = insertcols(C, L1+1, MARGIN);
C = C';
C = insertcols(C, L1+L2+1, MARGIN);
C = insertcols(C, L1+1, MARGIN);
C = C';

figure;
imshow(abs(C), [0 1]);
colorbar;
save_current_fig_as_pdf('100-NET12-DUMP/corr.pdf');
end

function [C] = insertcols(C, beforePos, width)
C2 = ones(size(C,1), width);
C1 = C(:, 1:beforePos-1);
C3 = C(:, beforePos:end);
C = [C1 C2 C3];
end