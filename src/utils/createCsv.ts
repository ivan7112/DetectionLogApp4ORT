
const createCsv= (data:Array<Object>) =>{

    const csvRows:Array<string>=[];

    const headers:string= Object.keys(data[0]).join(","); //Cabecera CSV

    csvRows.push(headers);

    for ( let i=0;i<data.length;i++){ 
        csvRows.push(Object.values(data[i]).join(","));

    }

    const csv:string=csvRows.join("\n");

    download(csv);

} 

export default createCsv;

const download= (data:string)=>{
    const blob=new Blob([data], {type:'text/csv'});
    const url=URL.createObjectURL(blob);

    
    const a=document.createElement('a');
    a.href=url;
    a.download='logGestures.csv';
    a.click();
}