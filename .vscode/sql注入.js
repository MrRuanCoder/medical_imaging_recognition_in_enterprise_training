str = ''
for (let i = 33; i < 127; i++) {
  s = String.fromCharCode(i)
  //防止SQL注入
  if (["'", '"', ';', '-', '/', '\\', '%', '_'].includes(s)) {
    str += '---------------------------------------------'
  }
  else
    str += s
}

console.log(str)