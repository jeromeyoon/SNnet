function vertex = genVertex( depth, grid, K )
%DEPTH2POINT Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    save = false;
end
sz = [size(depth) numel(depth)] ;

[x, y]  = meshgrid(1:sz(2),1:sz(1));
X = [x(:),y(:),ones(sz(3),1)]';
vertex = K\X;
vertex = vertex.*repmat(depth(:)',3,1);
vertex = vertex(:,grid);
