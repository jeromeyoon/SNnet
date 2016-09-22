clear all
close all

addpath('/research2/backup/work/jsonlab')
savepath = '/research2/ECCV_journal/with_light/json';
load('eccv_journal.mat');

trainidx = find(imdb.sets ==1);
testdx = find(imdb.sets ==2);


% traininput = cell2mat(imdb.dt(trainidx));
% traingt = cell2mat(imdb.gt(trainidx));
% testinput = cell2mat(imdb.dt(testdx));
% testgt = cell2mat(imdb.gt(testdx));

traininput1 =imdb.dt(trainidx);
traininput2 =imdb.light(trainidx);
traingt =imdb.gt(trainidx);
testinput1 =imdb.dt(testdx);
testinput2 = imdb.light(testdx);
testgt = imdb.gt(testdx);


assert(length(traininput1)==length(traininput2));
assert(length(traininput1)==length(traingt));

assert(length(testinput1)==length(testinput2));
assert(length(testinput1)==length(testgt));


traininput = cell2mat(traininput1);
trainlight = cell2mat(traininput2);
traingt = cell2mat(traingt);
testinput = cell2mat(testinput1);
testlight = cell2mat(testinput2);
testgt = cell2mat(testgt);

% num_traindata = length(traininput1);
% num_testinput = length(testinput1);

% step = 999;
% tmp_traininput=[];
% tmp_traingt=[];
% tmp_testnput=[];
% tmp_testgt=[];
% count =1;
% 
% for i=1:step:num_traindata
%     
%     t = step*count;
%      fprintf('%d/%d \n',i,t );
%     tmp = cell2mat(traininput(i:t));
%     tmp_traininput =cat(1,tmp_traininput,tmp);
%     
%     tmp2 = cell2mat(traingt(i:t));
%     tmp_traingt =cat(1,tmp_traingt,tmp2);
%     count = count+1;
% end
% traininput = tmp_trainput;
% traingt = tmp_traingt;
% 
% count =1;
% for i=1:step:num_testinput
%      t = step*count;
%     tmp = cell2mat(testinput(i:t));
%     tmp_testinput =cat(1,tmp_testinput,tmp);
%     
%     tmp2 = cell2mat(testgt(i:t));
%     tmp_testgt =cat(1,tmp_testgt,tmp2);
%     count = count+1;
% end
% testinput = tmp_testinput;
% testgt = tmp_testgt;


savejson('',traininput,fullfile(savepath,'traininput.json'));
savejson('',trainlight,fullfile(savepath,'trainlight.json'));
savejson('',traingt,fullfile(savepath,'traingt.json'));
savejson('',testinput,fullfile(savepath,'testinput.json'));
savejson('',testlight,fullfile(savepath,'testlight.json'));
savejson('',testgt,fullfile(savepath,'testgt.json'));

%load json file 
%loadjson('testinput_material.json')