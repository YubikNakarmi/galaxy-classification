document$.subscribe(() => {
  const NAV_KEY = 'sidebar-nav-hidden'
  const TOC_KEY = 'sidebar-toc-hidden'

  const navHidden = localStorage.getItem(NAV_KEY) === 'true'
  const tocHidden = localStorage.getItem(TOC_KEY) === 'true'

  document.body.classList.toggle('nav-hidden', navHidden)
  document.body.classList.toggle('toc-hidden', tocHidden)

  // Left toggle (primary sidebar / navigation)
  let leftBtn = document.querySelector('.sidebar-toggle.left')
  if (!leftBtn) {
    leftBtn = document.createElement('button')
    leftBtn.className = 'sidebar-toggle left'
    leftBtn.title = 'Toggle navigation'
    leftBtn.setAttribute('aria-label', 'Toggle navigation sidebar')
    leftBtn.onclick = () => {
      const isHidden = document.body.classList.toggle('nav-hidden')
      leftBtn.classList.toggle('collapsed', isHidden)
      leftBtn.setAttribute('aria-expanded', String(!isHidden))
      localStorage.setItem(NAV_KEY, String(isHidden))
    }
    document.body.appendChild(leftBtn)
  }
  leftBtn.classList.toggle('collapsed', navHidden)
  leftBtn.setAttribute('aria-expanded', String(!navHidden))

  // Right toggle (table of contents)
  let rightBtn = document.querySelector('.sidebar-toggle.right')
  if (!rightBtn) {
    rightBtn = document.createElement('button')
    rightBtn.className = 'sidebar-toggle right'
    rightBtn.title = 'Toggle table of contents'
    rightBtn.setAttribute('aria-label', 'Toggle table of contents')
    rightBtn.onclick = () => {
      const isHidden = document.body.classList.toggle('toc-hidden')
      rightBtn.classList.toggle('collapsed', isHidden)
      rightBtn.setAttribute('aria-expanded', String(!isHidden))
      localStorage.setItem(TOC_KEY, String(isHidden))
    }
    document.body.appendChild(rightBtn)
  }
  rightBtn.classList.toggle('collapsed', tocHidden)
  rightBtn.setAttribute('aria-expanded', String(!tocHidden))
})