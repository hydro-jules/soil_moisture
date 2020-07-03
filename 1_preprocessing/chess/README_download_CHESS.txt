STEPS to download CHESS data:

1)	Create a JASMIN account:
CHESS data is stored on JASMIN.
If you do not have a JASMIN account, you will need to:
- Create a JASMIN account: https://accounts.jasmin.ac.uk/services/group_workspaces/?query=hydro_jules 
- Set up everything on your machine to be able to access JASMIN: https://help.jasmin.ac.uk/article/189-get-started-with-jasmin
- Request access to the Group workspace hydro_jules: (https://accounts.jasmin.ac.uk/services/group_workspaces/?query=hydro_jules).

2)	Transfer the data using rsync:
Soil moisture data can be found in: 
/gws/nopw/j04/hydro_jules/data/uk/jules_outputs/chess/

If you need to copy the data to your local machine, follow these steps (on a Linux terminal):
eval   $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa_jasmin
#(you will be prompted for your JASMIN passphrase)
rsync -avzh <jasmin_username>@jasmin-xfer1.ceda.ac.uk:/gws/nopw/j04/hydro_jules/data/uk/jules_outputs/chess/<file_name> <path_on_your_local_machine>
NOTE: there is no space after <jasmin_username>@jasmin-xfer1.ceda.ac.uk:
