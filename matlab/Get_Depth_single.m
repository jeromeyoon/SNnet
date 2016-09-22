clear all;
close all;
clc;

s = 0.5;
pp = 9;
folder_set = {'011', '016', '021', '022', '033', '036', '038', '053', '059', '092'};
N_name = '12_Normal.bmp';
savetype = '.bmp';

M_name = 'mask.bmp';
%mainpath = '/research2/backup/work/PycharmProjects/ECCV2016/DCGAN_IR_single6/L1_ang_ei_loss_result';
mainpath = '/research2/ECCV_journal/with_light/NIR_single_MSE_ang_hinge/output';
maskpath = '/research2/IR_normal_small/mask';
savepath = '/home/yjyoon/Dropbox/ECCV_result/3D/with_light/';
if ~isdir(savepath)
    mkdir(savepath)
end
       
        N = im2double(imread(fullfile(mainpath,'input_006_006_001.png'))); 
        N = N.*2 -1;
        

        t = 1;
        im = im2double(imread(fullfile(mainpath,'input_006_006_001.png')));
        im = im(:,:,1); 

        %figure, imshow(im);

        % load mask image
        im_mask = sprintf('%s/%s/%d/%s',maskpath,folder_set{7},1,M_name);
        im_mask = imresize(imread(im_mask),s);
        im_mask = logical(im_mask);
        idx = find(im_mask==1);
        im_mask = repmat(im_mask,[1,1,3]);

        % im_mask = im;
        % im_mask(im~=0) = 1;

        N(im_mask==0) = 0;

 %%%%%% Invert normal to depth %%%%%%%%%%%%%%

        height = size(im,1);
        width = size(im,2);

        N = reshape(N,[height,width,3]);
        [x,y] = meshgrid(1:width,1:height);

        u = N(:,:,1);
        v = N(:,:,2);
        y = flipud(y);
        v = flipud(v);


        [X, Y, Z]=Depth_still(N,im_mask,height,width,3);

        Z = flipud(Z);



        K = [2.99e+03*s, 0, 7.99e+02*s ; 0, 2.99e+03*s, 5.99e+02*s ; 0,0,1];

        im_mask =im_mask(:,:,1);
        im_mask = flipud(im_mask);
        grid = logical(im_mask(:,:,1));
        vertex = genVertex (Z,grid,K);

        C = im2uint8(im);
        C = double(C);
        C= round(C);
        C = repmat(C,[1,1,3]);
        C = reshape(C,[height*width,3]);

        id = find(grid~=0);
        R = C(:,1);   G = C(:,2);   B = C(:,3);
        C_R=R(id);  C_G=G(id);  C_B=B(id);
        C_ = [C_R C_G C_B];
        C_=C_';

        pathresult = sprintf('%s/3D_%s_%03d%s',savepath,folder_set{1},uint8(1),savetype);
        % grid_ = flipud(grid);
        mesh = genMesh(vertex,grid);
        save_name = sprintf('%s%s%s.ply',pathresult,'_',num2str(s));
        saveMeshAsPly_still(save_name,vertex, mesh, [], C_);
