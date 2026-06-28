import qai_hub as hub

model_path = "models/depth_anything_v3.tflite"

job = hub.submit_profile_job(
    model=model_path,
    device=hub.Device("Samsung Galaxy S25 Ultra"),
)

print(job)