cloned rcam repo and had to install some stuff

```bash
git clone https://github.com/hu-po/rcam.git
sudo snap install rustup --classic
rustup update stable
sudo apt-get install -y libssl-dev libclang-dev
pip3 install rerun-sdk==0.23.2
echo "export PATH="/snap/bin:$PATH"" >> ~/.bashrc
cd rcam
cp ~/tatbot/.env .env
source .env
cargo clean && cargo build
target/debug/rcam capture-image --rerun
```

