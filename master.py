from datetime import datetime
import subprocess, os, shutil, time

name = "vk_viewer"

start_time = time.time()
release_name = "%s-%s" % (name, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

staging_dir = name
asset_dirs = ["models", "music", "scenes", "shaders", "textures"]

if os.path.exists(staging_dir):
	shutil.rmtree(staging_dir)

cargo_proc = subprocess.run([
		"cargo",
		"rustc",
		"--profile",
		"master",
		"--",
		"--cfg",
		"master"
	],
	check=True
)

#Copy redist DLLs
shutil.copytree("redist/", "%s/" % staging_dir)

os.mkdir("%s/data" % staging_dir)
for d in asset_dirs:
	if d == "models":
		os.mkdir("%s/data/%s" % (staging_dir, d))
		shutil.copy("data/%s/totoro_backup.glb" % d, "%s/data/%s/" % (staging_dir, d))
	else:
		shutil.copytree("data/%s" % d, "%s/data/%s" % (staging_dir, d))


shutil.copy("target/master/%s.exe" % name, "%s/" % staging_dir)
shutil.copy("run_from_cmd.bat", "%s/" % staging_dir)

#Compress the build into a zip archive
print("Compressing build to %s.zip..." % release_name)
shutil.make_archive(release_name, "zip", root_dir=staging_dir)

print("Done compressing in %.4f seconds" % (time.time() - start_time))

#Cleanup
#shutil.rmtree(staging_dir)