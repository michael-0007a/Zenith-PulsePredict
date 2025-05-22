let currentIndex = 0;
const slideInterval = 3000; // Interval in milliseconds (3 seconds)

function moveSlide() {
    const slides = document.querySelector('.testimonial-slide');
    const totalSlides = slides.children.length;

    currentIndex = (currentIndex + 1) % totalSlides;
    slides.style.transform = `translateX(-${currentIndex * 100}%)`;
}

// Start the auto-slide function
setInterval(moveSlide, slideInterval);


function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
}


(() => {
    'use strict'
  
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    const forms = document.querySelectorAll('.needs-validation')
  
    // Loop over them and prevent submission
    Array.from(forms).forEach(form => {
      form.addEventListener('submit', event => {
        if (!form.checkValidity()) {
          event.preventDefault()
          event.stopPropagation()
        }
  
        form.classList.add('was-validated')
      }, false)
    })
  })()