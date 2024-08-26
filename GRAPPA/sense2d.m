load('receive_slc90.mat')
load('mri.mat')
load('phantom3D_6coil.mat')
load('raw_data.mat')

addpath utils/


data = raw;
cal_imgs = ifft2c(data);
show_grid(abs(cal_imgs), [0, 16], gray)
N = ndims(raw);
sz = size(raw,N);
n = ceil(sqrt(sz));
m = ceil(sz/n);
idx = repmat({':'},1,N);


num_acs = 40;
kernel_size = [2,2];
eigen_thresh = 0.8;			% for mask size 

accel_factor = [4, 4];
imgs =  ifft2c(undersample2d(data, accel_factor));
show_grid(abs(imgs(:, : ,  1, :)), [0, 4], gray);

sense = coil_estimation_ESPIRIT(cal_imgs, 24, [6, 6], 0.8);
show_grid(abs(sense), [0, 1], jet);

recon = SENSE2d(imgs, sense, accel_factor);
disp(size(recon))
show_img(abs(recon), [0, 4], gray)

function recon=SENSE2d(imgs, sens, accel_factor)
    [Nx,Ny,Nz,Nc] = size(imgs);
    recon = zeros(Nx, Ny);
    r1 = accel_factor(1);
    r2 = accel_factor(2);
    for i = 1:Nx/r1
            for j = 1:Ny/r2
                C = [];
                for jj = 0:accel_factor-1
                    for ii = 0:accel_factor-1
                        C = [C; sens(i + (Nx/r1)*jj, j + (Ny/r2)*ii, 1, :)] ; 
                    end
                end
                C = transpose(squeeze(C));
                y = transpose(transpose ( squeeze(imgs(i, j, 1, :) )));
                x = pinv(C)*y;
                
                count = 1;
                for jj = 0:accel_factor-1
                    for ii = 0:accel_factor-1
                        
                        recon(i + jj*Nx/r1, j + ii*Ny/r2) = x(count);
                        count = count + 1;
                    end
                end
            end
    end    
end

function undersampled = undersample2d( kspace , accel_factor )
    r1 = accel_factor(1);
    r2 = accel_factor(2);
    undersampled = (kspace);
    for i = 1:size(kspace, 1)
        if(mod(i, r1) ~= 0)
            undersampled(i, :, :, : ) = zeros( size(undersampled(i, :, :, : )));
        end
    end
    for i = 1:size(kspace, 2)
        if(mod(i, r2) ~= 0)
            undersampled(:, i, :, : ) = zeros( size(undersampled(:, i, :, : )));
        end
    end
    [Nx,Ny,Nz,Nc] = size(kspace);
    r1 = accel_factor(1);
    r2 = accel_factor(2);

    undersampled = zeros(size(kspace));
    undersampled(1:r1:Nx, 1:r2:Ny, :, :) = kspace(1:r1:Nx, 1:r2:Ny, :, :);

end

function map=coil_estimation_ESPIRIT(raw_data, num_acs, kernel_size, eigen_thresh)
    

    img_patref = raw_data;
    receive = zeross(size(img_patref));
    tic
    parfor slc_select = 1:s(img_patref,3)     
        disp(num2str(slc_select))
        
        [maps, weights] = ecalib_soft( fft2c( sq(img_patref(:,:,slc_select,:)) ), num_acs, kernel_size, eigen_thresh );
    
        receive(:,:,slc_select,:) = permute(dot_mult(maps, weights >= eigen_thresh ), [1,2,4,3]);
    end 
    toc
    map = receive;

end


function img = coil_combine(imgs)
img  = sqrt(sum(abs(imgs).^2,4));
end

function sens = coil_estimation(imgs)
    [Nx,Ny,Nz,Nc] = size(imgs);
    m_aprox  = sqrt(sum(abs(imgs).^2,4));
    Cs1 = bsxfun(@rdivide, imgs, m_aprox);
    kern = ones(9) * (1/81);
    for i = 1:Nc
        Cs1(:, :, 1, i) = conv2( Cs1(:, :, 1, i), kern, "same");
    end
    threshold = 0.05*max(m_aprox(:));
    mask = abs(m_aprox) > threshold;
    mask = medfilt2(mask, [11, 11]);
    Cs2 = Cs1.*mask;
    sens = Cs2;
end


function recon = SENSE(imgs, sens, accel_factor)
    
    [Nx,Ny,Nz,Nc] = size(imgs);
    recon = zeros(Nx, Ny); 
    if accel_factor == 2
        for i = 1:Nx/2
            for j = 1:Ny
                C =  transpose(squeeze([sens(i, j, 1, :); sens(i + Nx/2, j, 1, :)])) ; 
                y = transpose(transpose ( squeeze(imgs(i, j, 1, :) )));
                x = pinv(C)*y;
                recon(i, j) = x(1);
                recon(i + Nx/2, j) = x(2);
            end
        end
    
    else
        for i = 1:Nx/accel_factor
            for j = 1:Ny
                C = [sens(i, j, 1, :)];
                for jj = 1:accel_factor-1
                    C = [C; sens(i + (Nx/accel_factor)*jj, j, 1, :)] ;                   
                end
                C = transpose(squeeze(C));
                
                y = transpose(transpose ( squeeze(imgs(i, j, 1, :) )));
                x = pinv(C)*y;

                for jj = 0:accel_factor-1
                    recon(i + jj*Nx/accel_factor, j) = x(jj+1);
                end

            end
        end
    end
    
end



function undersampled = undersample(kspace, accel_factor)
undersampled = zeros(size(kspace));
for i = 1:accel_factor:size(kspace, 1)-1
    undersampled(i, :, :, : ) = kspace(i, :, :, : );
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

%function out = fft2c(input)
%    out = fftshift(fft(ifftshift(input,1),[],1),1);
%    out = fftshift(fft(ifftshift(out,2),[],2),2);
%end

function out=ifft2c(input)
    out = fftshift(ifft(ifftshift(input,1),[],1),1);
    out = fftshift(ifft(ifftshift(out,2),[],2),2);
end