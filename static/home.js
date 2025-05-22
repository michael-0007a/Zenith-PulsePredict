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