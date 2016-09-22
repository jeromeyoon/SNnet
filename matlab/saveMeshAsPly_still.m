function saveMeshAsPly(name, vertex, face, normal, color)
nVertex = size(vertex, 2);

if ~isempty(face)
    nFaces = size(face, 2);
end
fid = fopen(name, 'w');

% Write a ply header
fprintf(fid, 'ply\n');
fprintf(fid, 'format binary_little_endian 1.0\n');
fprintf(fid, 'element vertex %lu\n', nVertex);
fprintf(fid, 'property float x\n');
fprintf(fid, 'property float y\n');
fprintf(fid, 'property float z\n');
if ~isempty(normal)
    fprintf(fid, 'property float nx\n');
    fprintf(fid, 'property float ny\n');
    fprintf(fid, 'property float nz\n');
end
if ~isempty(color)
    fprintf(fid, 'property uchar red\n');
    fprintf(fid, 'property uchar green\n');
    fprintf(fid, 'property uchar blue\n');
end

% fprintf(fid, 'element range_grid %d\n',nGrid);
% fprintf(fid, 'property list uchar int vertex_indices\n');

if ~isempty(face)
    fprintf(fid, 'element face %lu\n', nFaces);
    fprintf(fid, 'property list int int vertex_indices\n');
end
fprintf(fid, 'end_header\n');

vertex_ = [-vertex(1,:);vertex(2,:);-vertex(3,:)];
if ~isempty(normal)
    normal =  [normal(1,:);-normal(2,:);-normal(3,:)];
end

if (~isempty(normal) && ~isempty(color))
    for i = 1 : nVertex
        fwrite(fid, [vertex_(:,i); normal(:,i)], 'float');
        fwrite(fid, color(:,i), 'uint8');
    end
elseif ~isempty(color)
    for i = 1 : nVertex
        fwrite(fid, vertex_(:,i), 'float');
        fwrite(fid, color(:,i), 'uint8');
    end
elseif ~isempty(normal)
    fwrite(fid, [vertex_; normal], 'float');
else
    fwrite(fid, vertex_, 'float');
end

if ~isempty(face)
    fwrite(fid, [3*ones(1,nFaces); face - 1], 'int');
end

fclose(fid);

