document.getElementById('fileInput').addEventListener('change', function () {
  const preview = document.getElementById('preview');
  const file = this.files[0];
  preview.src = URL.createObjectURL(file);
});

function uploadImage() {
  const input = document.getElementById('fileInput');
  const file = input.files[0];
  const formData = new FormData();
  formData.append('image', file);

  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById('result').innerHTML = `Prediction: <strong>${data.result}</strong><br>Confidence: ${data.confidence}`;
    })
    .catch(err => {
      document.getElementById('result').innerText = 'Something went wrong.';
      console.error(err);
    });
}
