    // basic fetch for color update
    document.querySelectorAll('.color').forEach(colorElement => {
      colorElement.addEventListener('click', function() {
        const color = this.getAttribute('data-color');
        fetch(`http://localhost:8001/v1/color/${color}`)
          .then(response => response.json())
          .then(data => console.log(data))
          .catch(error => console.error('Error:', error));
      });
    });

    // fetch API level
    document.querySelectorAll('.level').forEach(levelElement => {
      levelElement.addEventListener('click', function() {
        const level = this.getAttribute('data-level');
        fetch(`http://localhost:8001/v1/${level}/`)
          .then(response => response.json())
          .then(data => console.log(data))
          .catch(error => console.error('Error:', error));
      });
    });

      // send filename to server
    document.querySelector('.button').addEventListener('click', function() {
      document.getElementById('fileInput').click();
    });
    document.getElementById('fileInput').addEventListener('change', function() {
      const file = this.files[0];
      console.log('Selected file:', file.name);
      fetch(`http://localhost:8001/v1/start/${file.name}`)
      .then(response => response.json())
      .then(() => alert("Data has been redacted successfully!"))
      .catch(error => console.error('Error:', error));

    });
