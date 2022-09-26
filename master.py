from datetime import datetime
import subprocess, os, shutil, time

name = "vk_viewer"

start_time = time.time()
release_name = "%s-%s" % (name, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

staging_dir = name
asset_dirs = ["data"]

if os.path.exists(staging_dir):
	shutil.rmtree(staging_dir)

cargo_proc = subprocess.run(["cargo", "build", "--release"], check=True)

#os.mkdir(staging_dir)

#Copy redist DLLs
shutil.copytree("redist/", "%s/" % staging_dir)

for d in asset_dirs:
	shutil.copytree(d, "%s/%s" % (staging_dir, d))


shutil.copy("target/release/%s.exe" % name, "%s/" % staging_dir)

#Compress the build into a zip archive
print("Compressing build...")
shutil.make_archive(release_name, "zip", root_dir=staging_dir)

print("Done compressing in %.4f seconds" % (time.time() - start_time))
start_time = time.time()

#Cleanup
shutil.rmtree(staging_dir)

archive_name = "%s.zip" % release_name
scp_proc = subprocess.run(["scp", archive_name, "pi@192.168.1.39:/home/pi/dist/%s" % archive_name], check=True)


print("Done uploading in %.4f seconds" % (time.time() - start_time))