function [face] = genMesh(vertex, grid)
sz = [size(grid) numel(grid)];
idx = reshape(1:sz(3), sz(1:2));
nvertex = size(vertex,2);
thresh_len = 30;
thresh_len2 = thresh_len^2;

% case 1 CCW
ltri11 = idx(1:end-1 ,1:end-1);
ltri12 = idx(2:end   ,1:end-1);
ltri13 = idx(1:end-1 ,2:end);
rtri11 = idx(2:end   ,2:end);
rtri12 = idx(1:end-1 ,2:end);
rtri13 = idx(2:end   ,1:end-1);

% case 2 CCW
ltri21 = idx(1:end-1 ,1:end-1);
ltri22 = idx(2:end   ,1:end-1);
ltri23 = idx(2:end   ,2:end);
rtri21 = idx(2:end   ,2:end);
rtri22 = idx(1:end-1 ,2:end);
rtri23 = idx(1:end-1 ,1:end-1);

% calculate distance
p1_idx  = [ltri12(:), ltri13(:)];
p2_idx  = [ltri21(:), ltri23(:)];
p_idx   = [p1_idx; p2_idx];

p1_mask = and(grid(p1_idx(:,1)), grid(p1_idx(:,2)));
p2_mask = and(grid(p2_idx(:,1)), grid(p2_idx(:,2)));
p_mask  = [p1_mask; p2_mask];
np1     = sum(p1_mask);

idx_grid = zeros(sz(1:2));
idx_grid(grid) = 1:nvertex;

p1 = vertex(:,idx_grid(p_idx(p_mask,1)));
p2 = vertex(:,idx_grid(p_idx(p_mask,2)));
d  = dist2(p1,p2);

d1 = d(1     :np1);
d2 = d(np1+1 :end);

d1_mat = zeros(sz(1:2)-1);
d2_mat = zeros(sz(1:2)-1);
d1_mat(p1_mask) = d1;
d2_mat(p2_mask) = d2;

cc = d1_mat < d2_mat;

vtri = [ltri11(cc), ltri12(cc), ltri13(cc);
        ltri21(~cc), ltri22(~cc), ltri23(~cc);
        rtri11(cc), rtri12(cc), rtri13(cc);
        rtri21(~cc), rtri22(~cc), rtri23(~cc)];
        
real_idx = zeros(sz(1:2));%reshape(cumsum(grid(:)), sz(1:2));
real_idx(grid) = 1:sum(grid(:));%real_idx = grid.*real_idx;
face   = [real_idx(vtri(:,1)), real_idx(vtri(:,2)), real_idx(vtri(:,3))]';
valid = face > 0;
valid = valid(1,:) & valid(2,:) & valid(3,:);
face   = face(:,valid);

%remove sliver faces
d12 = dist2(vertex(:,face(1,:)),vertex(:,face(2,:)));
d23 = dist2(vertex(:,face(2,:)),vertex(:,face(3,:)));
d31 = dist2(vertex(:,face(3,:)),vertex(:,face(1,:)));
face(:, (d12 > thresh_len2) | (d23 > thresh_len2) | (d31 > thresh_len2)) = [];

end

function d = dist2(v1,v2)
    d = sum((v1-v2).^2);
end

