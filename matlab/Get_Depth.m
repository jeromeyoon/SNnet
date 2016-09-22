% dataset_ = 'artificialLeather';
s = 0.5;
pp = 9;
folder_set = {'011', '016', '021', '022', '033', '036', '038', '053', '059', '092'};
N_name = '12_Normal.bmp';
savetype = '.bmp';
%N_name = 'masked_single_normal_L2ang.bmp';
%savetype = 'recon_L2ang.bmp';
M_name = 'mask.bmp';
%mainpath = '/research2/backup/work/PycharmProjects/ECCV2016/DCGAN_IR_single6/L1_ang_ei_loss_result';
mainpath = '/research2/ECCV_journal/with_light/NIR_single_L1_ang_2/output/1509/save';
maskpath = '/research2/IR_normal_small/mask';
savepath = '/home/yjyoon/Dropbox/ECCV_result/3D/hinge_L1_ang';
if ~isdir(savepath)
    mkdir(savepath)
end

for i = 1: 10 %objects 
    for j=1:9 % tilt angle
        fprintf('processing tilt angle %d/%d object %d/%d \n',j,9,i,10);
        
        
        %N_ = sprintf('%s/%s/%d/%s',mainpath,folder_set{i},j,N_name);
        N_ = dir(fullfile(sprintf('%s%s/%d/%s',mainpath,folder_set{i},j),'*.png'));
        N = im2double(imread(fullfile(sprintf('%s%s/%d/%s',mainpath,folder_set{i},j),N_(5).name))); 
        %N = imresize(N,s);
        N = N.*2 -1;
        % load image

        t = 1;
        im = im2double(imread(fullfile(sprintf('%s%s/%d/%s',mainpath,folder_set{i},j),N_(5).name))); 
        %im = imresize(im,s);
        %im = im2double(im); 
        im = im(:,:,1); 

        %figure, imshow(im);

        % load mask image
        im_mask = sprintf('%s/%s/%d/%s',maskpath,folder_set{i},j,M_name);
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
% figure
        u = N(:,:,1);
        v = N(:,:,2);
        y = flipud(y);
        v = flipud(v);
        % quiver(x,y,u,v)

        [X, Y, Z]=Depth_still(N,im_mask,height,width,3);

        Z = flipud(Z);

%         figure ('Name', '3D Reconstruction - View #1','NumberTitle','off'),
%         surfl(X,Y,Z)
%         shading interp
%         colormap (gray(256));
%         view(-180, -20);
%         title('Shape from shading');

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
% C_ = fliplr(C_);
% C_ = flipud(C_);


% SavePLY('test5.ply',vertex,C_);
        pathresult = sprintf('%s/3D_%s_%03d%s',savepath,folder_set{i},uint8(j),savetype);
        % grid_ = flipud(grid);
        mesh = genMesh(vertex,grid);
        save_name = sprintf('%s%s%s.ply',pathresult,'_',num2str(s));
        saveMeshAsPly_still(save_name,vertex, mesh, [], C_);
    end
end