const form=document.querySelector('#router-form');
const button=document.querySelector('#submit');
const error=document.querySelector('#error');

form.addEventListener('submit',async(event)=>{
  event.preventDefault(); error.hidden=true; button.disabled=true;
  button.querySelector('span').textContent='Routing…';
  try{
    const payload={prompt:document.querySelector('#prompt').value,max_input_price:+document.querySelector('#input-price').value,
      max_output_price:+document.querySelector('#output-price').value,requires_tools:document.querySelector('#tools').checked,
      requires_structured_output:document.querySelector('#structured').checked};
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json(); if(!response.ok)throw new Error(data.error||'Prediction failed');
    document.querySelector('.empty-state').hidden=true;document.querySelector('.result-content').hidden=false;
    document.querySelector('#result').classList.remove('empty');
    document.querySelector('#domain').textContent=data.domain.replaceAll('_',' ');
    document.querySelector('#tier').textContent=`Tier ${data.tier}`;
    document.querySelector('#model').textContent=data.selected_model;
    document.querySelector('#provider').textContent=data.provider;
    document.querySelector('#raw-tier').textContent=`Tier ${data.audit.llm_tier_prediction}`;
    document.querySelector('#ceiling').textContent=`$${payload.max_input_price} / $${payload.max_output_price}`;
    document.querySelector('#eligible').textContent=data.audit.eligible_models_after_constraints.length;
    document.querySelector('#reason').textContent=data.reason;
  }catch(err){error.textContent=err.message;error.hidden=false}
  finally{button.disabled=false;button.querySelector('span').textContent='Route request'}
});
