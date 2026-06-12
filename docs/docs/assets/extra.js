document$.subscribe(() => {
  // Left toggle
  if (!document.querySelector('.sidebar-toggle.left')) {
    const leftBtn = document.createElement('button')
    leftBtn.className = 'sidebar-toggle left'
    leftBtn.title = 'Toggle navigation'
    leftBtn.onclick = () => {
      document.body.classList.toggle('nav-hidden')
      leftBtn.classList.toggle('collapsed')
    }
    document.body.appendChild(leftBtn)
  }

  // Right toggle
  if (!document.querySelector('.sidebar-toggle.right')) {
    const rightBtn = document.createElement('button')
    rightBtn.className = 'sidebar-toggle right'
    rightBtn.title = 'Toggle table of contents'
    rightBtn.onclick = () => {
      document.body.classList.toggle('toc-hidden')
      rightBtn.classList.toggle('collapsed')
    }
    document.body.appendChild(rightBtn)
  }
})