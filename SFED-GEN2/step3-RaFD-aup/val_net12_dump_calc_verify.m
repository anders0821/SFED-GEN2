% 人脸验证roc auc

clc;
clear;
close all;
rng default

load 100-NET12-DUMP/dump.mat VAL_X VAL_Z_HAT VAL_H2_HAT;
VAL_X = double(VAL_X);
VAL_Z_HAT = double(VAL_Z_HAT);
VAL_H2_HAT = double(VAL_H2_HAT);

NUM_PAIR = 1000;
N = size(VAL_X, 4) / 7;

d = zeros(6, NUM_PAIR);
t = zeros(6, NUM_PAIR);
for i=1:NUM_PAIR
    % 生成样本对
    % 第一个人
    sub1 = randi(N);
    exp1 = randi(7);
    idx1 = (sub1-1)*7+exp1;

    % 同类
    sub2 = sub1;
    exp2 = randi(7-1);
    if(exp2>=exp1)
        exp2=exp2+1;
    end
    idx2 = (sub2-1)*7+exp2;

    % 不同类
    sub3 = randi(N-1);
    if(sub3>=sub1)
        sub3=sub3+1;
    end
    exp3 = randi(7);
    idx3 = (sub3-1)*7+exp3;

    % subplot(1,3,1);
    % imshow(squeeze(VAL_X(1,:,:,idx1)));
    % subplot(1,3,2);
    % imshow(squeeze(VAL_X(1,:,:,idx2)));
    % subplot(1,3,3);
    % imshow(squeeze(VAL_X(1,:,:,idx3)));
    % drawnow
    % pause(1)

    if(i<=NUM_PAIR/2)
        d(1,i) = pdist2(reshape(VAL_X(:,:,:,idx1), [1 64*64]), reshape(VAL_X(:,:,:,idx2), [1 64*64]), 'euclidean');
        d(2,i) = pdist2(reshape(VAL_X(:,:,:,idx1), [1 64*64]), reshape(VAL_X(:,:,:,idx2), [1 64*64]), 'cosine');
        d(3,i) = pdist2(reshape(VAL_Z_HAT(:,:,:,idx1), [1 64*64]), reshape(VAL_Z_HAT(:,:,:,idx2), [1 64*64]), 'euclidean');
        d(4,i) = pdist2(reshape(VAL_Z_HAT(:,:,:,idx1), [1 64*64]), reshape(VAL_Z_HAT(:,:,:,idx2), [1 64*64]), 'cosine');
        d(5,i) = pdist2(reshape(VAL_H2_HAT(:,idx1), [1 512]), reshape(VAL_H2_HAT(:,idx2), [1 512]), 'euclidean');
        d(6,i) = pdist2(reshape(VAL_H2_HAT(:,idx1), [1 512]), reshape(VAL_H2_HAT(:,idx2), [1 512]), 'cosine');
    else
        d(1,i) = pdist2(reshape(VAL_X(:,:,:,idx1), [1 64*64]), reshape(VAL_X(:,:,:,idx3), [1 64*64]), 'euclidean');
        d(2,i) = pdist2(reshape(VAL_X(:,:,:,idx1), [1 64*64]), reshape(VAL_X(:,:,:,idx3), [1 64*64]), 'cosine');
        d(3,i) = pdist2(reshape(VAL_Z_HAT(:,:,:,idx1), [1 64*64]), reshape(VAL_Z_HAT(:,:,:,idx3), [1 64*64]), 'euclidean');
        d(4,i) = pdist2(reshape(VAL_Z_HAT(:,:,:,idx1), [1 64*64]), reshape(VAL_Z_HAT(:,:,:,idx3), [1 64*64]), 'cosine');
        d(5,i) = pdist2(reshape(VAL_H2_HAT(:,idx1), [1 512]), reshape(VAL_H2_HAT(:,idx3), [1 512]), 'euclidean');
        d(6,i) = pdist2(reshape(VAL_H2_HAT(:,idx1), [1 512]), reshape(VAL_H2_HAT(:,idx3), [1 512]), 'cosine');
    end
end

% 可视化距离分布
m = min(d');
M = max(d');
subplot(3,2,1)
hist(d(1,:), 100)
title([m(1) M(1)])
subplot(3,2,2)
hist(d(2,:), 100)
title([m(2) M(2)])
subplot(3,2,3)
hist(d(3,:), 100)
title([m(3) M(3)])
subplot(3,2,4)
hist(d(4,:), 100)
title([m(4) M(4)])
subplot(3,2,5)
hist(d(5,:), 100)
title([m(5) M(5)])
subplot(3,2,6)
hist(d(6,:), 100)
title([m(6) M(6)])

% 绘制ROC
figure;
for i=1:6
    subplot(3,2,i)

    NUM_SAMPLE = 100;
    dots = zeros(2, NUM_SAMPLE);
    T = linspace(m(i), M(i)+1e-12, NUM_SAMPLE);
    for j = 1:NUM_SAMPLE
        t = T(j);
        groundtruth = [ones(1, NUM_PAIR/2) zeros(1, NUM_PAIR/2)];
        predict = double(d(i,:)<t);
        FPR = sum(predict & ~groundtruth)/sum(~groundtruth);
        TPR = sum(predict & groundtruth)/sum(groundtruth);
        dots(1, j) = FPR;
        dots(2, j) = TPR;
    end
    assert(dots(1,1)==0)
    assert(dots(2,1)==0)
    assert(dots(1,end)==1)
    assert(dots(2,end)==1)
    plot(dots(1, :), dots(2, :));
    
    AUC = 0;
    for j = 1:NUM_SAMPLE-1
        x1 = dots(1, j);
        x2 = dots(1, j+1);
        y1 = dots(2, j);
        y2 = dots(2, j+1);
        
        AUC = AUC + (y2+y1)*(x2-x1)/2;
    end
    title(AUC);
end

% 绘制ROC
figure;
subplot(2,2,3)
xM = 0;
ym = 1;
for i=[1 3 6]
    NUM_SAMPLE = 1000;
    dots = zeros(2, NUM_SAMPLE);
    T = linspace(m(i), M(i)+1e-12, NUM_SAMPLE);
    for j = 1:NUM_SAMPLE
        t = T(j);
        groundtruth = [ones(1, NUM_PAIR/2) zeros(1, NUM_PAIR/2)];
        predict = double(d(i,:)<t);
        FPR = sum(predict & ~groundtruth)/sum(~groundtruth);
        TPR = sum(predict & groundtruth)/sum(groundtruth);
        dots(1, j) = FPR;
        dots(2, j) = TPR;
    end
    assert(dots(1,1)==0)
    assert(dots(2,1)==0)
    assert(dots(1,end)==1)
    assert(dots(2,end)==1)
    
    plot(dots(1, :), dots(2, :));
    hold on;
    
    ym2 = min(dots(2, dots(1,:)>0));
    if(ym2<ym)
        ym = ym2;
    end
    xM2 = max(dots(1, dots(2,:)<1));
    if(xM2>xM)
        xM = xM2;
    end
end
legend('Euclidean distance in the space of X', 'Euclidean distance in the space of Z', 'Cosine distance in the space of H_2');
xlim([0, xM]);
ylim([ym, 1]);
xlabel('False positive rate');
ylabel('True positive rate');

save_current_fig_as_pdf('100-NET12-DUMP/verify.pdf')
