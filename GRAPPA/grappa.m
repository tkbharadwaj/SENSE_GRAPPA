%GRAPPA Implementation by Tejas Bharadwaj, 7.18.24

load('raw_data.mat');
[Nx,Ny,Nz,Nc] = size(raw);

imgs = ifft2c(raw);
show_grid(abs(imgs), [0 16], gray);
grappa_weights = ACS(raw, 10, 6); %raw data, nlines, accleeration rate

undersampled = undersample(raw, 6); %raw data, accleeration rate
naiive_recon = ifft2c(undersampled);
show_grid(abs(naiive_recon), [0 16], gray);
recon_kspace = GRAPPA(undersampled, grappa_weights, 6 ); %raw data, weights, accleeration rate
undersampled_recon = ifft2c( recon_kspace );
show_grid(abs(undersampled_recon), [0 16], gray);
show_img(coil_combine(undersampled_recon))


function undersampled = undersample(kspace, accel_factor)
%disp(size(kspace))
undersampled = zeros(size(kspace));
%disp(size(undersampled))
for i = 1:accel_factor:size(kspace, 1)-1
    undersampled(i, :, :, : ) = kspace(i, :, :, : );
end

end

function img = coil_combine(imgs)
img  = sqrt(sum(abs(imgs).^2,4));
end

function recon = GRAPPA(kspace, weights, accel_factor)
  if accel_factor == 2
    [Nx,Ny,Nz,Nc] = size(kspace);
    for i=2:Nx-1
       if mod(i-1, accel_factor) == 0
           %disp(kspace(i));
           continue      
       end
       source_points = [];
       for j=2:Ny-1
            pts = transpose( [squeeze(kspace(i-1, j-1:j+1, 1, :) ) ;  squeeze(kspace(i+1, j-1:j+1, 1, :) ) ] );
            pts = reshape(pts.', 1, []);
            source_points = [source_points; pts];
       end
       t = source_points*weights;
       %disp(size(t));
       t = reshape(t, [ 1 size(t, 1) 1 size(t, 2)]);
       kspace(i, 2:Ny-1, :, :) = t;
       recon = kspace;
    end
  else
    [Nx,Ny,Nz,Nc] = size(kspace);
    for kernel_num = 1:accel_factor-1
        for i=1:Nx-accel_factor
           if mod(i-1, accel_factor) ~= 0
               continue      
           end
           source_points = [];
           for j=2:Ny-1
             pts = transpose( [squeeze(kspace(i, j-1:j+1, 1, :) ) ;  squeeze(kspace(i+accel_factor, j-1:j+1, 1, :) ) ] );
             pts = reshape(pts.', 1, []);
             source_points = [source_points; pts];
           end
           t = source_points*(squeeze(weights(kernel_num, :, :, :)));
           t = reshape(t, [ 1, size(t, 1) ,1 ,size(t, 2)]);
           kspace(i+kernel_num, 2:Ny-1, :, :) = t;
           recon = kspace;
        end
    end
end
end


function w = ACS(kspace, nlines, accel_factor)
    if accel_factor == 2
    target_points = [];
    source_points = [];
    [Nx,Ny,Nz,Nc] = size(kspace);
    for i=(Nx/2 - nlines/2):(Nx/2 + nlines/2) 
        for j=2:Ny-1
            target_points = [target_points; transpose( squeeze(kspace(i, j, 1, :)) )];
            pts = transpose( [squeeze(kspace(i-1, j-1:j+1, 1, :) ) ;  squeeze(kspace(i+1, j-1:j+1, 1, :) ) ] );
            pts = reshape(pts.', 1, []);
            source_points = [source_points; pts];
        end
    end
    w = pinv(source_points)*target_points;
    
    else
        [Nx,Ny,Nz,Nc] = size(kspace);
        w = zeros(accel_factor-1, Nc*6,Nc)
        for kernel_num = 1:accel_factor-1
            target_points = [];
            source_points = [];
            for i=(Nx/2 - nlines/2):(Nx/2 + nlines/2) 
                for j=2:Ny-1
                    target_points = [target_points;  transpose(squeeze(kspace(i+kernel_num, j, 1, :)))    ];
                    pts = transpose( [squeeze(kspace(i, j-1:j+1, 1, :) ) ;  squeeze(kspace(i+accel_factor, j-1:j+1, 1, :) ) ] );
                    pts = reshape(pts.', 1, []);
                    source_points = [source_points; pts];
                end
            end
        w(kernel_num, :, : ) = pinv(source_points)*target_points;
        end
    end
end

%helper functions taken from https://github.com/mchiew/SENSE-tutorial/blob/main/SENSE_tutorial.m
function show_grid(data, cscale, cmap)
    if nargin < 2
        cscale = [];
    end
    if nargin < 3
        cmap = gray;
    end
    figure();
    N = ndims(data);
    sz = size(data,N);
    n = ceil(sqrt(sz));
    m = ceil(sz/n);
    idx = repmat({':'},1,N);
    for i = 1:m
        for j = 1:n
            idx{N} = (i-1)*m+j;
            subplot('position',[(i-1)/m (n-j)/n (1/m-0.005) (1/n-0.005)]);
            imshow(data(idx{:}),cscale,'colormap',cmap);
        end
    end
end

function show_img(data, cscale, cmap)
   if nargin < 2 || isempty(cscale)
       cscale = [-inf inf];
   end
   if nargin < 3
       cmap = gray;
   end
   figure();
   imagesc(data);
   axis equal
   colormap(cmap);
   clim(cscale);
   plotH = gca;
   plotH.XTick = [];plotH.YTick = [];plotH.YColor = 'w';plotH.XColor = 'w';
end

function out = fft2c(input)
    out = fftshift(fft(ifftshift(input,1),[],1),1);
    out = fftshift(fft(ifftshift(out,2),[],2),2);
end

function out = ifft2c(input)
    out = fftshift(ifft(ifftshift(input,1),[],1),1);
    out = fftshift(ifft(ifftshift(out,2),[],2),2);
end