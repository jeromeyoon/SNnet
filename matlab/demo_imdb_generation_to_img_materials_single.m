
function demo_imdb_generation_to_img_materials_single(varargin)
mainPath = '/research2/IR_normal_small/save%03d/' ;
lightPath = '/research2/IR_normal_small/Light_source/' ;
savePath = '/research2/IR_normal_small/journal/save%03d/%03d/' ;

% factor = [0.484196608471131,0.460990462318154,0.550586230467512, ...
%           0.345559851780990,0.404826544147166,1,0.754078953071583,0.933102428363664, ...
%           0.275101498901340,0.401835345608246,0.505235539098123,0.430413905646977];
% for i =1:101
%     for j=1:9
%          savePath_a = sprintf(savePath,i,j) ;
%          rmdir(savePath_a,'s');
%     end
% end


%1 to 101
sub_Path = '%d/' ;
%1 to 9
img_Format = '*.bmp' ;
sve_dt_Format = '%03d.png' ;
sve_light_Format = '%03d_light.png' ;
sve_gt_Format = '%03d_gt.png' ;

%val_s = 1 ;
sample=[1,2,3,4,5,6,7,8,9,10,11,12,13];
val_s = [11,36,16,33,38,21,22,92,53,59];
sets = ones(1,101) ;
sets(val_s) = 2 ;
sets_all = [] ;
Dt_path = [];
Gt_path = [];
light_path=[];
light_files = dir(fullfile(lightPath,'*.png'));
%light_files =light_files(4:end-3);

lights = [] ;
for i = 1:12
   tmp = imread(fullfile(lightPath,light_files(i).name)); 
     lights = cat(3,lights,tmp);
end

chk_List = zeros(101,9) ;
for i = 1 : 101,
    for j = 1 : 9,
        
        [im,~] = demo_mydir([sprintf([mainPath,sub_Path],i,j),img_Format]) ;
        
        %1 : 10, exceptation
        if length(im) ~= 13, chk_List(i,j) = 1 ; continue ; end;
        im = sort({im.name});
        im  = natsort(im);
        imo = [] ;
        for k = 1 : length(im),
            if find(sample ==k)
                tmp = imread(im{k});
                %tmp = imresize(imread(im{k}),0.5);
                imo = cat(3,imo,tmp);
            %if k < 13,
            %    tmp = imread(im{k});
                %tmp = imresize(imread(im{k}),[600,800]);
                %tmp = single(imresize(imread(im{k}),[600,800]))/factor(k);
            %    imo = cat(3,imo,tmp);
            %else
            %    tmp = imread(im{k});
            %    imo = cat(3,imo,tmp);
            end
        end % 1to3 : groundtruth , 4to6 : 4,6,9 (training) 
        %if size(imo,3) ~= 15, chk_List(i,j) = 2 ; continue ; end;
        
        mask = im2bw(imo(:,:,end-2:end),0.05) ;
        
        step_slide = 64;%104;
        size_slide = 224 ;
        h_slide = floor((size(mask,1)-size_slide)/step_slide) ; 
        w_slide = floor((size(mask,2)-size_slide)/step_slide) ;
%         [ww,hh] = meshgrid(1:size(mask,2),1:size(mask,1));

        savePath_a = sprintf(savePath,i,j) ;
        if ~exist(savePath_a,'dir'), mkdir(savePath_a) ; end ;
        IU  =  1 ;
        for m = 1 : h_slide,
            for n = 1 : w_slide,
                hs = (m-1)*step_slide + 1 ;
                ws = (n-1)*step_slide + 1 ;
                he = hs+size_slide-1 ;
                we = ws+size_slide-1 ;
               
                loc_mask  = mask(hs:he,ws:we,:) ;
                if nnz(loc_mask)/numel(loc_mask) < 0.95,
                    continue; 
                end;
                
                dt_loc = imo(hs:he,ws:we,:) ;
                light_loc = lights(hs:he,ws:we,:);
                
                for in = 1 : 12, %IR, (1:12) / gt, (13:15)
                    s= (in-1)*3+1;
                    e = in*3;
                    imwrite(uint8(dt_loc(:,:,in)),...
                        sprintf([savePath_a,sve_dt_Format],IU)) ;
                    imwrite(uint8(light_loc(:,:,s:e)),...
                        sprintf([savePath_a,sve_light_Format],IU)) ;
                    
                    if in == 1,
                     gtPathSave =...
                         sprintf([savePath_a,sve_gt_Format],IU);
                     imwrite(uint8( dt_loc(:,:,end-2:end)),...
                        gtPathSave) ;
                    end
                    Dt_path = [Dt_path;{sprintf([savePath_a,sve_dt_Format],IU)}] ;
                    Gt_path = [Gt_path;{gtPathSave}] ;
                    light_path = [light_path;{sprintf([savePath_a,sve_light_Format],IU)}] ;
                    sets_all = [sets_all, sets(i)];
                    IU = IU + 1 ;
                    
                end
            end
        end
        fprintf('%d materials (%d) =%f.. \n',i,101, j/9) ;
    end
end

imdb.dt = Dt_path ;
imdb.gt = Gt_path ;

imdb.light = light_path ;
imdb.sets = sets_all ;
save('eccv_journal.mat','imdb') ;

save('chk_List_224_light.mat','chk_List') ;
        
        
        
        
