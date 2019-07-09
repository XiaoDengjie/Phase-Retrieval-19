function IMG=GetSPEC(FileToOpen,FileIndex)
%usage example:  IMAGE=GetAnImageDirect(2239,1)
%imagesc(IMAGE)


%FileOffset=find(DUMP.FILEID==FileToOpen,1,'first')

%%i_dataname0='Camera::FrameV1/XrayTransportDiagnostic.0:Opal1000.0';
%modify the data name to your data
i_dataname0='Camera::FrameV1/FeeHxSpectrometer.0:OrcaFl40.0';
%FileToOpen = 'r2777';
run=num2str(double(FileToOpen));
h5_filename=['diamcc14-r',run,'.h5'];

IMID=FileIndex;
%disp(['File: ',run, ' - IMID: ',num2str(IMID)])

% [~,y_size0]=readIPMdata(0,i_dataname0,h5_filename,II+1,0);
% i_size0=[0,y_size0-1]; %Number of data for images
% [i_time0,~]=readIPMdata(0,i_dataname0,h5_filename,II+1,i_size0);
fina = h5_filename;
stepNO = 1;
hdf5scsteppath = ['/Configure:0000/Run:0000/CalibCycle:' num2str(stepNO-1,'%04d') '/'];

dataset0 = [hdf5scsteppath,i_dataname0,'/image'];


file = H5F.open (fina, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
dset0 = H5D.open (file, dataset0);
dataspace0 = H5D.get_space(dset0);

slabsize=1;

memspaceID0 = H5S.create_simple(length(slabsize), slabsize, slabsize);

H5S.select_hyperslab(dataspace0, 'H5S_SELECT_SET', IMID-1, [], 1, []);

IMG = transpose(H5D.read(dset0, 'H5ML_DEFAULT', memspaceID0, dataspace0, 'H5P_DEFAULT'));
IMG=double(IMG(end:-1:1,end:-1:1));

filename_im = [num2str(FileToOpen) ,'-', num2str(FileIndex), 'SPECresult.png'];
filename_dat = [num2str(FileToOpen) , '-', num2str(FileIndex), 'SPECfile.csv'];


csvwrite(join(filename_dat), IMG); %added to save intensity values of image
img = imagesc(IMG);

saveas(img, join(filename_im)); %added to save image



