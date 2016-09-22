%This code to computing scale invariance[From David Eigen NIPS 2014]

clear all
%close all
clc

NIR = im2double(imread('/research2/IR_normal_small/save011/1/1.bmp'));
Surface = im2double(imread('/research2/IR_normal_small/save011/1/12_Normal.bmp'));
Surface = (Surface)*2.0 -1.0;
Light = im2double(imread('/research2/IR_normal_small/Light_source/L01.png'));
load('L.mat');
%NIR_normal = normal(NIR);
Surface_normal = normal(Surface);
%Light_normal = normal(Light);
Light_normal = L(1,:);
Light_normal = repmat(Light_normal,1200*1600,1);
Light_normal = reshape(Light_normal,1200,1600,3);
NIR_hat = sum(Surface_normal.* Light_normal,3);
%normal_light(L);
scale_loss = scaleinvariance(NIR_hat,NIR);

function output = normal(input_)
    normal2 = input_.^2;
    normal2 = repmat(sum(normal2,3),1,1,3);
    normal2 = sqrt(normal2);
    normal2(normal2 ==0) = exp(-10);
    output = input_./normal2;

end


function output = scaleinvariance(input_,GT)
    input_ = (input_+1.0)/2.0;
    GT = (GT+1.0)/2.0;
    GT(GT==0) = exp(-10);
    input_(input_==0) = exp(-10);
    diff =   log(input_) -log(GT);
    output1 = sum(sum((diff.^2)))/(size(input_,1)*size(input_,2));
    output2 = sum(sum(diff)).^2/(size(input_,1)*size(input_,2))^2;
    output = output1-output2;


end

function  normal_light(input_)

    %-1~1 to 0~1
    L_hat = (input_+1.0)/2.0;
    for i =1:12
        L_ = L_hat(i,:);
        L_ =repmat(L_,1200*1600,1);
        L_ = reshape(L_,1200,1600,3);
        imwrite(L_,sprintf('L_%03d.png',i));
    end
    
end