# vogon2 - a benchmark runner

Run system. Run benchmarks. Collect results. Collect environment. Compile reports. Leverage containers.

## Setup
See requirements.txt and ansible/roles/packages/tasks/main.yaml

## One Shot Mode

Example:

```bash
vogon2.py test \
  --under-test-image quay.io/s3gw/s3gw:latest \  # s3gw image. will be pulled
  --suite demo \ # benchmark suite to run
  --archive-dir /tmp \  # place to archive files created during testing (e.g warp csv.zst files)
  --storage-device /dev/disk/by-id/ata-HL-DT-ST_DVDRAM_GH24NSC0_K85F4RG1218 \  # device to get information from
  --storage-partition /dev/loop0 \  # device / partition / loopback file to mkfs and mount
  --mountpoint /home/vogon/mnt \  # filesystem created with mkfs mounted here
  --docker-api unix:/run/podman/podman.sock \  # docker and podman(docker compt API) work
  --sqlite /home/vogon/vogon2.sqlite3 \
  --mkfs=/sbin/mkfs.ext4
```
See commandline help for all parameters.

## Scheduler

Using the included simple scheduler, benchmark runs can be queued up in advanced.
The scheduler watches a 'todo' directory for job files. When found it runs vogon with system-wide plus per-job settings.
Once done it moves the file to a 'done' or 'failed' directory.

Example:
```
sched.py --debug run  --config-file sched.json --sched-dir sched
```

See sched/sched.json_example for config file options.
