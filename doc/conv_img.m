% PSYCH 221 COURSE PROJECT
% This program takes in jpeg images and first makes the RGB to  XYZ
% conversion. Then assuming either the L cone or the M cone is missing, it
% deletes information related to the cone of interest and then reconverts
% it into the RGB space. This image roughly represents the normal image
% perceived by a dichromat. Then, subtracting this from the original image,
% we find the information lost when the image is seen by a dichromat. We
% then make a transformation on the error function so as to map it to
% something that could be perceived by a dichromat.
% For that we take the red component of the information, which is likely to
% be lost the most, and then using some weigth function add it to the blue
% and the red. Finally we add this new modified error function to the
% original image. When this image goes through the vischeck algorithm, now
% previously invisible details become visible.

clear;
file_name = 'grandparents' ;
%transorm matrices
lms2lmsp = [0 2.02344 -2.52581; 0 1 0; 0 0 1] ;
lms2lmsd = [1 0 0; 0.494207 0 1.24827; 0 0 1] ;
lms2lmst = [1 0 0; 0 1 0; -0.395913 0.801109 0] ;

rgb2lms = [17.8824 43.5161 4.11935; 3.45565 27.1554 3.86714; 0.0299566 0.184309 1.46709] ;
lms2rgb = inv(rgb2lms) ;

%read picture into RGB value



RGB = double(imread(file_name,'jpeg'));
sizeRGB = size(RGB) ;

%transform to LMS space
for i = 1:sizeRGB(1)
    for j = 1:sizeRGB(2)
        rgb = RGB(i,j,:);
        rgb = rgb(:);
        
        LMS(i,j,:) = rgb2lms * rgb;
    end
end

%transform to colorblind LMS values
for i = 1:sizeRGB(1)
    for j = 1:sizeRGB(2)
        lms = LMS(i,j,:);
        lms = lms(:);
        
        LMSp(i,j,:) = lms2lmsp * lms;
        LMSd(i,j,:) = lms2lmsd * lms;
    end
end

%transform new LMS value to RGB values
for i = 1:sizeRGB(1)
    for j = 1:sizeRGB(2)
        lmsp = LMSp(i,j,:);
        lmsp = lmsp(:);

        lmsd = LMSd(i,j,:);
        lmsd = lmsd(:);

        RGBp(i,j,:) = lms2rgb * lmsp;
        RGBd(i,j,:) = lms2rgb * lmsd;
      
    end
end

%calculate errors between two RGB values
errorp = (RGB-RGBp) ;
errord = (RGB-RGBd) ;

%daltonize (modifying errors)
err2mod = [0 0 0; .7 1 0; .7 0 1];
for i = 1:sizeRGB(1)
    for j = 1:sizeRGB(2)
        err = errorp(i,j,:);
        err = err(:);
        
        ERR(i,j,:) = err2mod * err;
    end
end

dtnp = ERR + RGB;

%convert to uint8
dtnp = uint8(dtnp);
ERR = uint8(ERR);
errorp = uint8(errorp);
errord = uint8(errord);
RGBp = uint8(RGBp);
RGBd = uint8(RGBd);

%write to file
imwrite(RGBp,[file_name 'p.jpeg'],'jpeg');
imwrite(RGBd,[file_name 'd.jpeg'],'jpeg');
%imwrite(ERR,[file_name 'err.jpeg'],'jpeg');
%imwrite(errorp,[file_name 'errp.jpeg'],'jpeg');
%imwrite(errord,[file_name 'errd.jpeg'],'jpeg');
imwrite(dtnp,[file_name '_dtn.jpeg'],'jpeg');
