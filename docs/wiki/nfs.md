# NFS Setup

on server `rpi2`:

```bash
sudo apt install nfs-kernel-server
sudo mkdir -p /home/rpi2/tatbot/nfs
sudo chmod 777 /home/rpi2/tatbot/nfs
sudo nano /etc/exports
# add this line:
> /home/rpi2/tatbot/nfs 192.168.1.0/24(rw,sync,no_subtree_check)
sudo exportfs -ra
sudo systemctl restart nfs-server
sudo exportfs -v
# enable on startup
sudo systemctl enable nfs-server
```

on client `rpi1`, `trossen-ai`, `ook`, `ojo`, and `oop`:

```bash
sudo apt install nfs-common
showmount -e 192.168.1.99
mkdir -p ~/tatbot/nfs
sudo mount -t nfs 192.168.1.99:/home/rpi2/tatbot/nfs ~/tatbot/nfs
# enable on startup
sudo nano /etc/fstab
# add this line:
> 192.168.1.99:/home/rpi2/tatbot/nfs /home/<USERNAME>/tatbot/nfs nfs defaults,nolock,vers=3,_netdev 0 0
sudo systemctl daemon-reload
sudo mount -a
```